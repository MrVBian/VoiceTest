#include "sensevoice.h"

SenseVoice::SenseVoice(QObject *parent) : QObject(parent)
{
    setupAudioFormat();
}

SenseVoice::~SenseVoice()
{
    stopRecording();
}

void SenseVoice::setupAudioFormat()
{
    format.setSampleRate(targetSampleRate);
    format.setChannelCount(targetChannelCount);
    format.setSampleFormat(QAudioFormat::Int16);
}

void SenseVoice::startRecording()
{
    if (recording) return;

    QAudioDevice device = QMediaDevices::defaultAudioInput();
    if (device.isNull()) {
        qWarning() << "No audio input device available";
        return;
    }

    // 检查设备是否支持所需格式
    if (!device.isFormatSupported(format)) {
        qWarning() << "Default format not supported, using nearest";
        format = device.preferredFormat();
        qWarning() << "Device does not support 16kHz mono, fallback to:"
                   << format.sampleRate() << "Hz,"
                   << format.channelCount() << "ch.";
    }

    // 保存实际使用的采样率和通道数
    actualSampleRate = format.sampleRate();
    actualChannelCount = format.channelCount();

    audioSource = new QAudioSource(device, format, this);
    audioIODevice = audioSource->start();

    if (!audioIODevice) {
        qWarning() << "Failed to start audio input";
        return;
    }

    connect(audioIODevice, &QIODevice::readyRead, this, &SenseVoice::handleDataReady);

    audioBuffer.clear();
    tempBuffer.clear();
    recording = true;
    emit recordingStatusChanged(true);
}

void SenseVoice::stopRecording()
{
    if (!recording) return;

    if (audioSource) {
        audioSource->stop();
        disconnect(audioIODevice, nullptr, this, nullptr);
        delete audioSource;
        audioSource = nullptr;
    }

    recording = false;
    emit recordingStatusChanged(false);
}

bool SenseVoice::isRecording() const
{
    return recording;
}

void SenseVoice::handleDataReady()
{
    // 读取所有可用数据
    QByteArray newData = audioIODevice->readAll();
    tempBuffer.append(newData);

    // 计算20ms对应的样本数
    const int samplesPer20ms = actualSampleRate / 50; // 50 = 1000ms/20ms
    const int bytesPer20ms = samplesPer20ms * actualChannelCount * sizeof(qint16);

    // 处理完整的20ms片段
    while (tempBuffer.size() >= bytesPer20ms) {
        QByteArray frame = tempBuffer.left(bytesPer20ms);
        tempBuffer.remove(0, bytesPer20ms);

        // 转换为目标格式
        const qint16* samples = reinterpret_cast<const qint16*>(frame.constData());
        int totalSamples = frame.size() / sizeof(qint16);
        QVector<qint16> pcm(samples, samples + totalSamples);

        QVector<qint16> converted = convertTo16kMono(pcm, actualSampleRate, actualChannelCount);

        // 存储转换后的数据
        audioBuffer.enqueue(QByteArray(reinterpret_cast<const char*>(converted.data()),
                                       converted.size() * sizeof(qint16)));
    }
}

QVector<qint16> SenseVoice::convertTo16kMono(const QVector<qint16> &input, int inRate, int inChannels)
{
    // 如果已经是目标格式，直接返回
    if (inRate == targetSampleRate && inChannels == targetChannelCount) {
        return input;
    }

    // 1. 转换为单声道
    QVector<qint16> mono;
    if (inChannels > 1) {
        int frames = input.size() / inChannels;
        mono.reserve(frames);

        for (int i = 0; i < frames; i++) {
            qint32 sum = 0;
            for (int c = 0; c < inChannels; c++) {
                sum += input[i * inChannels + c];
            }
            mono.append(static_cast<qint16>(sum / inChannels));
        }
    } else {
        mono = input;
    }

    // 2. 重采样到16kHz
    if (inRate != targetSampleRate) {
        double ratio = static_cast<double>(targetSampleRate) / inRate;
        int outFrames = static_cast<int>(mono.size() * ratio);
        QVector<qint16> out;
        out.reserve(outFrames);

        for (int i = 0; i < outFrames; i++) {
            double srcPos = i / ratio;
            int idx = static_cast<int>(srcPos);
            double frac = srcPos - idx;

            if (idx >= mono.size() - 1) {
                out.append(mono.last());
            } else {
                qint16 s1 = mono[idx];
                qint16 s2 = mono[idx + 1];
                qint16 interpolated = static_cast<qint16>(s1 + (s2 - s1) * frac);
                out.append(interpolated);
            }
        }
        return out;
    }

    return mono;
}

void SenseVoice::writeWavHeader(QFile &file, quint32 dataSize)
{
    // RIFF块
    file.write("RIFF", 4);
    quint32 chunkSize = 36 + dataSize; // 36是除了数据块之外的大小
    qToLittleEndian(chunkSize, reinterpret_cast<uchar*>(&chunkSize));
    file.write(reinterpret_cast<const char*>(&chunkSize), 4);

    // WAVE标识
    file.write("WAVE", 4);

    // fmt子块
    file.write("fmt ", 4);
    quint32 subChunk1Size = 16; // PCM格式块大小
    qToLittleEndian(subChunk1Size, reinterpret_cast<uchar*>(&subChunk1Size));
    file.write(reinterpret_cast<const char*>(&subChunk1Size), 4);

    quint16 audioFormat = 1; // PCM
    qToLittleEndian(audioFormat, reinterpret_cast<uchar*>(&audioFormat));
    file.write(reinterpret_cast<const char*>(&audioFormat), 2);

    quint16 numChannels = 1; // 单声道
    qToLittleEndian(numChannels, reinterpret_cast<uchar*>(&numChannels));
    file.write(reinterpret_cast<const char*>(&numChannels), 2);

    quint32 sampleRate = 16000; // 采样率
    qToLittleEndian(sampleRate, reinterpret_cast<uchar*>(&sampleRate));
    file.write(reinterpret_cast<const char*>(&sampleRate), 4);

    quint32 byteRate = 16000 * 1 * 2; // 每秒字节数 = 采样率 * 通道数 * 位深度/8
    qToLittleEndian(byteRate, reinterpret_cast<uchar*>(&byteRate));
    file.write(reinterpret_cast<const char*>(&byteRate), 4);

    quint16 blockAlign = 1 * 2; // 每个样本的字节数 = 通道数 * 位深度/8
    qToLittleEndian(blockAlign, reinterpret_cast<uchar*>(&blockAlign));
    file.write(reinterpret_cast<const char*>(&blockAlign), 2);

    quint16 bitsPerSample = 16; // 位深度
    qToLittleEndian(bitsPerSample, reinterpret_cast<uchar*>(&bitsPerSample));
    file.write(reinterpret_cast<const char*>(&bitsPerSample), 2);

    // data子块
    file.write("data", 4);
    qToLittleEndian(dataSize, reinterpret_cast<uchar*>(&dataSize));
    file.write(reinterpret_cast<const char*>(&dataSize), 4);
}

void SenseVoice::saveToWav()
{
    if (audioBuffer.isEmpty()) {
        qWarning() << "No audio data to save";
        return;
    }

    // 合并所有缓冲数据
    QByteArray audioData;
    while (!audioBuffer.isEmpty()) {
        audioData.append(audioBuffer.dequeue());
    }

    // 写入WAV文件
    QFile file("recording.wav");
    if (!file.open(QIODevice::WriteOnly)) {
        qWarning() << "Failed to create WAV file";
        return;
    }

    writeWavHeader(file, audioData.size());
    file.write(audioData);
    file.close();

    qDebug() << "Audio saved to recording.wav, size:" << audioData.size() << "bytes";
}

void SenseVoice::saveRecording()
{
    if (!recording) {
        qWarning() << "Not recording, cannot save";
        return;
    }

    // 先停止录音
    stopRecording();

    // 处理缓冲区残留数据
    if (tempBuffer.size() > 0) {
        const qint16* samples = reinterpret_cast<const qint16*>(tempBuffer.constData());
        int totalSamples = tempBuffer.size() / sizeof(qint16);
        QVector<qint16> pcm(samples, samples + totalSamples);

        QVector<qint16> converted = convertTo16kMono(pcm, actualSampleRate, actualChannelCount);

        audioBuffer.enqueue(QByteArray(reinterpret_cast<const char*>(converted.data()),
                                       converted.size() * sizeof(qint16)));
        tempBuffer.clear();
    }

    saveToWav();
}
