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
    connect(m_timer, &QTimer::timeout, this, &AudioCapture::processAudioData);
    m_totalBytesProcessed = 0; // 在构造函数中初始化


    // 初始化wav头
    initFixedHeader();

    // Vad
    if (SherpaOnnxFileExists("./vad/silero_vad.onnx")) {
        printf("Use silero-vad\n");
        vad_filename = "./vad/silero_vad.onnx";
        use_silero_vad = 1;
    } else if (SherpaOnnxFileExists("./vad/ten-vad.onnx")) {
        printf("Use ten-vad\n");
        vad_filename = "./vad/ten-vad.onnx";
        use_ten_vad = 1;
    } else {
        fprintf(stderr, "Please provide either silero_vad.onnx or ten-vad.onnx\n");
        return;
    }

    memset(&vadConfig, 0, sizeof(vadConfig));

    if (use_silero_vad) {
        vadConfig.silero_vad.model = vad_filename;
        vadConfig.silero_vad.threshold = 0.25;
        vadConfig.silero_vad.min_silence_duration = 0.5;
        vadConfig.silero_vad.min_speech_duration = 0.5;
        vadConfig.silero_vad.max_speech_duration = 10;
        vadConfig.silero_vad.window_size = 512;
    } else if (use_ten_vad) {
        vadConfig.ten_vad.model = vad_filename;
        vadConfig.ten_vad.threshold = 0.25;
        vadConfig.ten_vad.min_silence_duration = 0.5;
        vadConfig.ten_vad.min_speech_duration = 0.5;
        vadConfig.ten_vad.max_speech_duration = 10;
        vadConfig.ten_vad.window_size = 256;
    }

    vadConfig.sample_rate = 16000;
    vadConfig.num_threads = 1;
    vadConfig.debug = 1;

    vad = SherpaOnnxCreateVoiceActivityDetector(&vadConfig, 30);

    if (vad == NULL) {
        fprintf(stderr, "Please check your recognizer config!\n");
        return ;
    }
}

AudioCapture::~AudioCapture()
{
    stopCapture();
    SherpaOnnxDestroyVoiceActivityDetector(vad);
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
    m_timer->start(32); // 20ms间隔
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
        rawData = resampleTo16kHzMono(rawData, m_audioFormat);
    }

    if (!rawData.isEmpty()) {
        m_audioQueue.enqueue(rawData);
        m_totalBytesProcessed += rawData.size(); // 更新总字节数
        qDebug() << "Processed remaining data:" << rawData.size() << "bytes";
    }
}

void AudioCapture::processAudioData()
{
    if (!m_audioIO) return;

    // 计算20ms音频数据所需字节数（使用实际采样率）
    // sampleRate: 48000
    // bytesPerFrame: 4(2通道*2字节)
    // bytesNeeded: 3840
    // rawData: 3840/640

    const int kBufferDurationMs = 32;
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
        rawData = resampleTo16kHzMono(rawData, m_audioFormat);
        // qDebug() << "PCM: " << rawData.size();
    }

    bool voiceDetected = false;
    if (!rawData.isEmpty()) {
        try{
            // rawData 是 int16_t PCM，16kHz单通道
            int numSamples = rawData.size() / sizeof(int16_t);
            if (numSamples <= 0) return;

            const int16_t* pcm = reinterpret_cast<const int16_t*>(rawData.constData());
            std::vector<float> floatSamples(numSamples);
            for (int i = 0; i < numSamples; ++i) {
                floatSamples[i] = pcm[i] / 32768.0f; // int16 -> float [-1, 1]
            }
            if (!vad) {
                qWarning() << "VAD is not initialized!";
                return;
            }

            // 直接送入 VAD
            SherpaOnnxVoiceActivityDetectorAcceptWaveform(vad, floatSamples.data(), numSamples);
            // SherpaOnnxVoiceActivityDetectorFlush(vad);


            while (!SherpaOnnxVoiceActivityDetectorEmpty(vad)) {
                const SherpaOnnxSpeechSegment *segment =
                    SherpaOnnxVoiceActivityDetectorFront(vad);

                voiceDetected = true;
                float start = segment->start / 16000.0f;
                float duration = segment->n / 16000.0f;
                float stop = start + duration;

                qDebug() << QString("Voice detected: %1   start=%2, stop=%3")
                                .arg(voiceDetected)    // %1
                                .arg(start, 0, 'f', 3) // %2 (格式化为浮点数，保留3位小数)
                                .arg(stop, 0, 'f', 3); // %3

                SherpaOnnxDestroySpeechSegment(segment);
                SherpaOnnxVoiceActivityDetectorPop(vad);
            }
            // SherpaOnnxFreeWave(wave);


            m_audioQueue.enqueue(rawData);
            m_totalBytesProcessed += rawData.size(); // 更新总字节数

            // 调试输出
            qDebug() << "Processed chunk:" << rawData.size() << "bytes"
                     << "Total:" << m_totalBytesProcessed << "bytes";
        }
        catch (const std::exception& e) {
            qDebug() << "Exception in VAD processing:" << e.what();
            return;
        }
    }
}

QByteArray AudioCapture::resampleTo16kHzMono(const QByteArray &input, const QAudioFormat &format)
{
    // 输入参数
    const int inSampleRate = format.sampleRate();
    const int outSampleRate = 16000;
    const int channels = format.channelCount();
    const int bytesPerSample = format.bytesPerSample();
    const int kBufferDurationMs = 32;

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
    const int sampleRate = 16000;
    const int channels = 1;
    const int bitsPerSample = 16;
    const int byteRate = sampleRate * channels * bitsPerSample / 8;
    const int blockAlign = channels * bitsPerSample / 8;

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

    // WAV参数
    const int sampleRate = 16000;
    const int channels = 1;
    const int bitsPerSample = 16;
    const int byteRate = sampleRate * channels * bitsPerSample / 8;
    const int blockAlign = channels * bitsPerSample / 8;

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
