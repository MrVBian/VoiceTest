#ifndef AUDIOCAPTURE_H
#define AUDIOCAPTURE_H

#include <QObject>
#include <QAudioSource>
#include <QAudioFormat>
#include <QBuffer>
#include <QQueue>
#include <QFile>
#include <QTimer>
#include <QAudioDevice>
#include <c-api.h>

class VoiceData{
public:
    std::pair<float, float> time;
    QString context;
    VoiceData(const std::pair<float, float>& t, const QString& c) : time(t), context(c) {}
};

class AudioCapture : public QObject
{
    Q_OBJECT
public:
    explicit AudioCapture(QObject *parent = nullptr);
    ~AudioCapture();

    void startCapture();
    void stopCapture();
    QVector<VoiceData> voiceData;

public:signals:
    void errorOccurred(const QString &message);
    void voiceDataSend(const VoiceData& data);

private slots:
    void processAudioData();

private:
    QAudioSource *m_audioSource = nullptr;
    QIODevice *m_audioIO = nullptr;
    QTimer *m_timer = nullptr;
    QAudioFormat m_audioFormat;
    QQueue<QByteArray> m_audioQueue;
    bool m_resampleRequired = false;
    qint64 m_totalBytesProcessed = 0; // 确保这里声明了成员变量

    void setupAudioFormat();
    QByteArray resampleTo16kHzMono(const QByteArray &input, const QAudioFormat &format);
    void writeWavFile();
    void processRemainingData();



    QByteArray fixedHeader; // 存储除dataSize外的固定头部数据
    qint32 fixedHeaderSize; // 固定头部的大小（不包括RIFF块大小和data块大小）
    void initFixedHeader();
    QByteArray createHeader(qint64 dataSize);

    std::vector<float> vadBuffer;  // 缓存用于VAD的浮点数据

    // Vad
    const SherpaOnnxVoiceActivityDetector *vad;
    SherpaOnnxVadModelConfig vadConfig;
    const char *vad_filename;
    int32_t use_silero_vad = 0;
    int32_t use_ten_vad = 0;


    SherpaOnnxOfflineParaformerModelConfig paraformer_config;
    SherpaOnnxOfflineModelConfig offline_model_config;
    const SherpaOnnxOfflineRecognizer *recognizer;


    const int sampleRate = 16000;
    const int channels = 1;
    const int bitsPerSample = 16;
    const int byteRate = sampleRate * channels * bitsPerSample / 8;
    const int blockAlign = channels * bitsPerSample / 8;
    // 计算32ms音频数据所需字节数（使用实际采样率）
    const int kBufferDurationMs = 32;
};

#endif // AUDIOCAPTURE_H
