#include "audiocapture.h"
#include <QAudioDevice>
#include <QMediaDevices>
#include <QDebug>
#include <QtEndian>
#include <QElapsedTimer>


AudioCapture::AudioCapture(QObject *parent) : QObject(parent)
{
    setupAudioFormat();
    m_timer = new QTimer(this);
    m_timer->setInterval(kBufferDurationMs);
    connect(m_timer, &QTimer::timeout, this, &AudioCapture::processAudioData);
    m_totalBytesProcessed = 0; // 在构造函数中初始化


    // 初始化wav头
    initFixedHeader();

    // Vad
    if (SherpaOnnxFileExists("./vad/silero_vad.onnx")) {
        printf("Use silero-vad\n");
        vad_filename = "./vad/silero_vad.onnx";
        use_silero_vad = 1;
    } else {
        fprintf(stderr, "Please provide either silero_vad.onnx or ten-vad.onnx\n");
        return;
    }

    memset(&vadConfig, 0, sizeof(vadConfig));

    // Silero VAD 配置参数
    vadConfig.silero_vad.model = vad_filename;
    vadConfig.silero_vad.threshold = 0.3;           // 语音活动检测的阈值，范围[0,1]，值越小对语音越敏感
    vadConfig.silero_vad.min_silence_duration = 0.2; // 最小静音持续时间（秒），短于此时间的静音会被忽略
    vadConfig.silero_vad.min_speech_duration = 0.2;  // 最小语音持续时间（秒），短于此时间的语音段会被过滤
    vadConfig.silero_vad.max_speech_duration = 10;   // 最大单段语音持续时间（秒），用于限制单次语音输入长度
    vadConfig.silero_vad.window_size = 512;          // 分析窗口大小（采样点数），影响VAD的时间分辨率

    // 音频处理基础配置
    vadConfig.sample_rate = 16000;  // 音频采样率（Hz），通常使用16kHz用于语音处理
    vadConfig.num_threads = 2;      // 处理线程数，1表示单线程处理
    vadConfig.debug = 0;            // 调试模式开关，1开启调试信息输出，0关闭

    vad = SherpaOnnxCreateVoiceActivityDetector(&vadConfig, 30);

    if (vad == NULL) {
        fprintf(stderr, "Please check your recognizer config!\n");
        return ;
    }



    // Paraformer config
    const char *model_filename =
        "sherpa-onnx-paraformer-zh-small/model.int8.onnx";
    const char *tokens_filename =
        "sherpa-onnx-paraformer-zh-small/tokens.txt";
    const char *provider = "cpu";
    memset(&paraformer_config, 0, sizeof(paraformer_config));
    paraformer_config.model = model_filename;

    // Offline model config
    memset(&offline_model_config, 0, sizeof(offline_model_config));
    offline_model_config.debug = 0;
    offline_model_config.num_threads = 2;
    offline_model_config.provider = provider;
    offline_model_config.tokens = tokens_filename;
    offline_model_config.paraformer = paraformer_config;

    // Recognizer config
    SherpaOnnxOfflineRecognizerConfig recognizer_config;
    memset(&recognizer_config, 0, sizeof(recognizer_config));
    recognizer_config.decoding_method = "greedy_search";
    recognizer_config.model_config = offline_model_config;

    recognizer = SherpaOnnxCreateOfflineRecognizer(&recognizer_config);
}

AudioCapture::~AudioCapture()
{
    stopCapture();
    SherpaOnnxDestroyVoiceActivityDetector(vad);
    SherpaOnnxDestroyOfflineRecognizer(recognizer);
}

void AudioCapture::setupAudioFormat()
{
    m_audioFormat.setSampleRate(16000);
    m_audioFormat.setChannelCount(1);
    m_audioFormat.setSampleFormat(QAudioFormat::Int16);

    // Check if device supports 16kHz mono
    QAudioDevice device = QMediaDevices::defaultAudioInput();
    if (!device.isFormatSupported(m_audioFormat)) {
        qDebug() << "16kHz mono not supported. Using default format.";
        m_audioFormat = device.preferredFormat();

        // 确保使用16位整数格式
        if (m_audioFormat.sampleFormat() != QAudioFormat::Int16) {
            qDebug() << "Forcing Int16 sample format";
            m_audioFormat.setSampleFormat(QAudioFormat::Int16);
        }

        m_resampleRequired = (m_audioFormat.sampleRate() != 16000 ||
                              m_audioFormat.channelCount() != 1);
    }
}

void AudioCapture::startCapture()
{
    if (m_audioSource) return;
    voiceData.clear();

    QAudioDevice device = QMediaDevices::defaultAudioInput();
    m_audioSource = new QAudioSource(device, m_audioFormat, this);
    m_audioIO = m_audioSource->start();

    if (!m_audioIO) {
        emit errorOccurred("Failed to start audio capture");
        return;
    }

    // 重置计数器
    m_totalBytesProcessed = 0; // 确保这里使用了成员变量

    // 使用更精确的定时器
    m_timer->start();
    qDebug() << "Capture started with format:"
             << "\nSample rate:" << m_audioFormat.sampleRate()
             << "\nChannels:" << m_audioFormat.channelCount()
             << "\nSample format:" << m_audioFormat.sampleFormat()
             << "\nResampling:" << (m_resampleRequired ? "Yes" : "No");
}

void AudioCapture::stopCapture()
{
    if (m_timer && m_timer->isActive()) {
        m_timer->stop();
    }

    // 处理剩余数据
    if (m_audioIO && m_audioSource) {
        processRemainingData();
    }

    if (m_audioSource) {
        m_audioSource->stop();
        delete m_audioSource;
        m_audioSource = nullptr;
        m_audioIO = nullptr;
    }

    if (!m_audioQueue.isEmpty()) {
        writeWavFile();
        m_audioQueue.clear();
    }

    qDebug() << "Total audio data processed:" << m_totalBytesProcessed << "bytes";
}

// 实现处理剩余数据的函数
void AudioCapture::processRemainingData()
{
    if (!m_audioIO) return;

    // 获取缓冲区中剩余的所有数据
    qint64 bytesAvailable = m_audioIO->bytesAvailable();
    if (bytesAvailable <= 0) return;

    QByteArray rawData = m_audioIO->readAll();
    if (rawData.isEmpty()) return;

    // 如果需要重采样
    if (m_resampleRequired) {
        qDebug() << "Resampling... Original size:" << rawData.size();
        rawData = resampleTo16kHzMono(rawData, m_audioFormat);
        qDebug() << "Resampled size:" << rawData.size();
    }

    if (rawData.isEmpty())
        return;

    try{
        int numSamples = rawData.size() / sizeof(int16_t);
        const int16_t* pcm = reinterpret_cast<const int16_t*>(rawData.constData());
        std::vector<float> floatSamples(numSamples);
        for (int i = 0; i < numSamples; ++i) {
            floatSamples[i] = pcm[i] / 32768.0f; // int16 -> float [-1, 1]
        }

        // 直接送入 VAD
        SherpaOnnxVoiceActivityDetectorAcceptWaveform(vad, floatSamples.data(), numSamples);
        SherpaOnnxVoiceActivityDetectorFlush(vad);

        while (!SherpaOnnxVoiceActivityDetectorEmpty(vad)) {
            const SherpaOnnxSpeechSegment *segment =
                SherpaOnnxVoiceActivityDetectorFront(vad);

            float start = segment->start / 16000.0f;
            float duration = segment->n / 16000.0f;
            float stop = start + duration;

            if (recognizer == NULL) {
                fprintf(stderr, "Please check your config!\n");
            }
            else {
                const SherpaOnnxOfflineStream *stream =
                    SherpaOnnxCreateOfflineStream(recognizer);

                try{
                    SherpaOnnxAcceptWaveformOffline(stream, sampleRate, segment->samples,
                                                    segment->n);
                    SherpaOnnxDecodeOfflineStream(recognizer, stream);
                    const SherpaOnnxOfflineRecognizerResult *result = SherpaOnnxGetOfflineStreamResult(stream);

                    auto tmp = VoiceData(std::make_pair(start, stop), result->text);
                    emit voiceDataSend(tmp);
                    voiceData.append(tmp);
                    // QString context = QString("%1-%2  Decoded text: %s").arg(start, 0, 'f', 3).arg(stop, 0, 'f', 3).arg(result->text);

                    SherpaOnnxDestroyOfflineRecognizerResult(result);
                }
                catch (const std::exception& e) {
                    qDebug() << "Exception in VAD processing:" << e.what();
                }

                SherpaOnnxDestroyOfflineStream(stream);
            }

            SherpaOnnxDestroySpeechSegment(segment);
            SherpaOnnxVoiceActivityDetectorPop(vad);
        }
        m_audioQueue.enqueue(rawData);
        m_totalBytesProcessed += rawData.size(); // 更新总字节数

        // 调试输出
        qDebug() << "Processed remaining data:" << rawData.size() << "bytes";

    }
    catch (const std::exception& e) {
        qDebug() << "Exception in VAD processing:" << e.what();
        return;
    }
}

void AudioCapture::processAudioData()
{
    if (!m_audioIO) return;

    int bytesPerFrame = m_audioFormat.bytesPerFrame();
    int bytesNeeded = (m_audioFormat.sampleRate() * bytesPerFrame * kBufferDurationMs) / 1000;
    bytesNeeded = qMax(bytesNeeded, 0);

    // 确保有足够数据可用，最后小于kBufferDurationMs数据会被丢弃
    if (m_audioIO->bytesAvailable() < bytesNeeded) {
        return;
    }

    QByteArray rawData = m_audioIO->read(bytesNeeded);
    if (rawData.isEmpty()) return;
    // qDebug() << QString("sampleRate:%1   bytesNeeded=%2  rawData=%3")
    //                 .arg(m_audioFormat.sampleRate())
    //                 .arg(bytesNeeded)
    //                 .arg(rawData.size());
    // 如果需要重采样
    if (m_resampleRequired) {
        // qDebug() << "Resampling... Original size:" << rawData.size();
        rawData = resampleTo16kHzMono(rawData, m_audioFormat);
        // qDebug() << "Resampled size:" << rawData.size();  // 应为512*sizeof(int16_t)=1024
    }

    if (rawData.isEmpty()){
        return;
    }

    try{
        // rawData 是 int16_t PCM，16kHz单通道
        int numSamples = rawData.size() / sizeof(int16_t);
        const int16_t* pcm = reinterpret_cast<const int16_t*>(rawData.constData());
        std::vector<float> floatSamples(numSamples);
        for (int i = 0; i < numSamples; ++i) {
            floatSamples[i] = pcm[i] / 32768.0f; // int16 -> float [-1, 1]
        }

        // 直接送入 VAD
        SherpaOnnxVoiceActivityDetectorAcceptWaveform(vad, floatSamples.data(), numSamples);
        // qDebug() << "TEST: " << !SherpaOnnxVoiceActivityDetectorEmpty(vad);

        while (!SherpaOnnxVoiceActivityDetectorEmpty(vad)) {
            const SherpaOnnxSpeechSegment *segment =
                SherpaOnnxVoiceActivityDetectorFront(vad);

            float start = segment->start / 16000.0f;
            float duration = segment->n / 16000.0f;
            float stop = start + duration;

            if (recognizer == NULL) {
                fprintf(stderr, "Please check your config!\n");
            }
            else {
                const SherpaOnnxOfflineStream *stream =
                    SherpaOnnxCreateOfflineStream(recognizer);

                try{
                    SherpaOnnxAcceptWaveformOffline(stream, sampleRate, segment->samples,
                                                    segment->n);
                    SherpaOnnxDecodeOfflineStream(recognizer, stream);
                    const SherpaOnnxOfflineRecognizerResult *result = SherpaOnnxGetOfflineStreamResult(stream);

                    auto tmp = VoiceData(std::make_pair(start, stop), result->text);
                    emit voiceDataSend(tmp);
                    voiceData.append(tmp);
                    // QString context = QString("%1-%2  Decoded text: %s").arg(start, 0, 'f', 3).arg(stop, 0, 'f', 3).arg(result->text);

                    SherpaOnnxDestroyOfflineRecognizerResult(result);
                }
                catch (const std::exception& e) {
                    qDebug() << "Exception in VAD processing:" << e.what();
                }

                SherpaOnnxDestroyOfflineStream(stream);
            }


            SherpaOnnxDestroySpeechSegment(segment);
            SherpaOnnxVoiceActivityDetectorPop(vad);
        }
        m_audioQueue.enqueue(rawData);
        m_totalBytesProcessed += rawData.size(); // 更新总字节数

        // // 调试输出
        // qDebug() << "Processed chunk:" << rawData.size() << "bytes"
        //          << "Total:" << m_totalBytesProcessed << "bytes";
    }
    catch (const std::exception& e) {
        qDebug() << "Exception in VAD processing:" << e.what();
        return;
    }
}

QByteArray AudioCapture::resampleTo16kHzMono(const QByteArray &input, const QAudioFormat &format)
{
    // 输入参数
    const int inSampleRate = format.sampleRate();
    const int outSampleRate = sampleRate;
    const int channels = format.channelCount();
    const int bytesPerSample = format.bytesPerSample();

    // 验证输入格式
    if (bytesPerSample != sizeof(qint16)) {
        qWarning() << "Unsupported sample size:" << bytesPerSample;
        return input;
    }

    // 计算输入样本数
    const int inSamples = input.size() / (channels * bytesPerSample);
    if (inSamples < 2) {
        qWarning() << "Not enough samples for resampling:" << inSamples;
        return QByteArray();
    }

    // 计算输出样本数（kBufferDurationMs 毫秒的目标）
    const int outSamples = (outSampleRate * kBufferDurationMs) / 1000;

    QByteArray output;
    output.resize(outSamples * sizeof(qint16));

    const qint16 *inPtr = reinterpret_cast<const qint16*>(input.constData());
    qint16 *outPtr = reinterpret_cast<qint16*>(output.data());

    // 改进的重采样算法
    const double ratio = static_cast<double>(inSampleRate) / outSampleRate;
    double pos = 0.0;

    for (int i = 0; i < outSamples; i++) {
        // 计算输入位置
        int idx = static_cast<int>(pos);
        double frac = pos - idx;

        // 确保索引在范围内
        if (idx >= inSamples - 1) {
            idx = inSamples - 2;
            frac = 1.0;
        }

        // 对所有通道取平均（转换为单声道）
        qint32 sum = 0;
        for (int ch = 0; ch < channels; ch++) {
            qint16 sample1 = inPtr[(idx * channels) + ch];
            qint16 sample2 = inPtr[((idx + 1) * channels) + ch];

            // 线性插值
            qint16 interpolated = static_cast<qint16>(
                sample1 + frac * (sample2 - sample1)
                );

            sum += interpolated;
        }

        // 写入输出
        *outPtr++ = static_cast<qint16>(sum / channels);
        pos += ratio;
    }

    return output;
}

void AudioCapture::initFixedHeader() {
    fixedHeader.clear();
    fixedHeader.reserve(36); // 固定部分大小：44-8=36字节

    // WAVE标识和fmt块（固定部分）
    fixedHeader.append("WAVEfmt ");

    // fmt块内容
    qint32 fmtSize = 16;
    fixedHeader.append(reinterpret_cast<const char*>(&fmtSize), 4);
    qint16 audioFormat = 1; // PCM
    fixedHeader.append(reinterpret_cast<const char*>(&audioFormat), 2);
    qint16 numChannels = channels;
    fixedHeader.append(reinterpret_cast<const char*>(&numChannels), 2);
    qint32 sampleRate32 = sampleRate;
    fixedHeader.append(reinterpret_cast<const char*>(&sampleRate32), 4);
    qint32 byteRate32 = byteRate;
    fixedHeader.append(reinterpret_cast<const char*>(&byteRate32), 4);
    qint16 blockAlign16 = blockAlign;
    fixedHeader.append(reinterpret_cast<const char*>(&blockAlign16), 2);
    qint16 bitsPerSample16 = bitsPerSample;
    fixedHeader.append(reinterpret_cast<const char*>(&bitsPerSample16), 2);

    // data块标识（固定部分）
    fixedHeader.append("data");

    // 固定头部大小 = 当前长度 + 4（为dataSize预留位置）
    fixedHeaderSize = fixedHeader.size() + 4;
}

QByteArray AudioCapture::createHeader(qint64 dataSize) {
    QByteArray header;
    header.clear();
    header.reserve(44);

    // RIFF头（动态部分）
    header.append("RIFF");
    qint32 fileSize = static_cast<qint32>(dataSize + fixedHeaderSize + 8); // 8是"RIFF"和fileSize自身
    header.append(reinterpret_cast<const char*>(&fileSize), 4);

    // 添加固定部分
    header.append(fixedHeader);

    // 添加data块大小（动态部分）
    qint32 dataSize32 = static_cast<qint32>(dataSize);
    header.append(reinterpret_cast<const char*>(&dataSize32), 4);

    return header;
}

void AudioCapture::writeWavFile()
{
    QFile file("captured_audio.wav");
    if (!file.open(QIODevice::WriteOnly)) {
        emit errorOccurred("Failed to create WAV file");
        return;
    }

    // 计算总数据大小
    qint64 dataSize = 0;
    for (const QByteArray &chunk : std::as_const(m_audioQueue)) {
        dataSize += chunk.size();
    }

    // 创建WAV头
    QByteArray header = createHeader(dataSize);

    // 写入头
    file.write(header);

    // 写入音频数据
    for (const QByteArray &chunk : std::as_const(m_audioQueue)) {
        file.write(chunk);
    }

    file.close();

    // 计算预期时长
    double expectedDuration = static_cast<double>(dataSize) / (sampleRate * channels * (bitsPerSample / 8));
    qDebug() << "Audio saved to captured_audio.wav ("
             << dataSize << "bytes, "
             << expectedDuration << "seconds)";
}
