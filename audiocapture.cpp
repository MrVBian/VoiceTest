#include "audiocapture.h"
#include <QAudioDevice>
#include <QMediaDevices>
#include <QDebug>
#include <QtEndian>
#include <cmath> // 添加cmath用于数学计算

AudioCapture::AudioCapture(QObject *parent) : QObject(parent)
{
    setupAudioFormat();
    m_timer = new QTimer(this);
    m_timer->setInterval(20); // 20ms processing interval
    connect(m_timer, &QTimer::timeout, this, &AudioCapture::processAudioData);
}

AudioCapture::~AudioCapture()
{
    stopCapture();
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

    if (m_audioSource) {
        m_audioSource->stop();
        delete m_audioSource;
        m_audioSource = nullptr;
    }

    if (!m_audioQueue.isEmpty()) {
        writeWavFile();
        m_audioQueue.clear();
    }
}

void AudioCapture::processAudioData()
{
    if (!m_audioIO) return;

    // 计算20ms音频数据所需字节数
    int bytesPerSample = m_audioFormat.bytesPerSample();
    int bytesPerFrame = m_audioFormat.bytesPerFrame();
    int bytesNeeded = (16000 * bytesPerFrame * 20) / 1000; // 始终以16kHz为目标

    // 确保有足够数据可用
    if (m_audioIO->bytesAvailable() < bytesNeeded) {
        return;
    }

    QByteArray rawData = m_audioIO->read(bytesNeeded);
    if (rawData.isEmpty()) return;

    // 如果需要重采样
    if (m_resampleRequired) {
        rawData = resampleTo16kHzMono(rawData, m_audioFormat);
    }

    m_audioQueue.enqueue(rawData);
}

QByteArray AudioCapture::resampleTo16kHzMono(const QByteArray &input, const QAudioFormat &format)
{
    // 输入参数
    const int inSampleRate = format.sampleRate();
    const int outSampleRate = 16000;
    const int channels = format.channelCount();
    const int bytesPerSample = format.bytesPerSample();

    // 验证输入格式
    if (bytesPerSample != sizeof(qint16)) {
        qWarning() << "Unsupported sample size:" << bytesPerSample;
        return input;
    }

    // 计算输入输出样本数
    const int inSamples = input.size() / (channels * bytesPerSample);
    const int outSamples = (inSamples * outSampleRate) / inSampleRate;

    QByteArray output;
    output.resize(outSamples * sizeof(qint16));

    const qint16 *inPtr = reinterpret_cast<const qint16*>(input.constData());
    qint16 *outPtr = reinterpret_cast<qint16*>(output.data());

    // 简单但有效的重采样算法
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
    for (const QByteArray &chunk : m_audioQueue) {
        dataSize += chunk.size();
    }

    // 创建WAV头
    QByteArray header;
    header.reserve(44);

    // RIFF头
    header.append("RIFF");
    qint32 fileSize = static_cast<qint32>(dataSize + 36);
    header.append(reinterpret_cast<const char*>(&fileSize), 4);

    // WAVE标识
    header.append("WAVE");

    // fmt块
    header.append("fmt ");
    qint32 fmtSize = 16;
    header.append(reinterpret_cast<const char*>(&fmtSize), 4);
    qint16 audioFormat = 1; // PCM
    header.append(reinterpret_cast<const char*>(&audioFormat), 2);
    qint16 numChannels = channels;
    header.append(reinterpret_cast<const char*>(&numChannels), 2);
    qint32 sampleRate32 = sampleRate;
    header.append(reinterpret_cast<const char*>(&sampleRate32), 4);
    qint32 byteRate32 = byteRate;
    header.append(reinterpret_cast<const char*>(&byteRate32), 4);
    qint16 blockAlign16 = blockAlign;
    header.append(reinterpret_cast<const char*>(&blockAlign16), 2);
    qint16 bitsPerSample16 = bitsPerSample;
    header.append(reinterpret_cast<const char*>(&bitsPerSample16), 2);

    // data块
    header.append("data");
    qint32 dataSize32 = static_cast<qint32>(dataSize);
    header.append(reinterpret_cast<const char*>(&dataSize32), 4);

    // 写入头
    file.write(header);

    // 写入音频数据
    for (const QByteArray &chunk : m_audioQueue) {
        file.write(chunk);
    }

    file.close();
    qDebug() << "Audio saved to captured_audio.wav (" << dataSize << "bytes)";
}
