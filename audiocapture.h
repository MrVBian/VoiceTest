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

class AudioCapture : public QObject
{
    Q_OBJECT
public:
    explicit AudioCapture(QObject *parent = nullptr);
    ~AudioCapture();

    void startCapture();
    void stopCapture();

signals:
    void errorOccurred(const QString &message);

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


    // Vad
    const char *vad_filename;
    int32_t use_silero_vad = 0;
    int32_t use_ten_vad = 0;
};

#endif // AUDIOCAPTURE_H
