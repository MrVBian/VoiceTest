#ifndef SENSEVOICE_H
#define SENSEVOICE_H

extern "C" {
    #include <sherpa-onnx/c-api/c-api.h>
}

#include <QObject>
#include <QAudioSource>
#include <QAudioDevice>
#include <QMediaDevices>
#include <QBuffer>
#include <QQueue>
#include <QFile>
#include <QDebug>
#include <QtEndian>
#include <algorithm>
#include <cmath>

class SenseVoice : public QObject
{
    Q_OBJECT
public:
    explicit SenseVoice(QObject *parent = nullptr);
    ~SenseVoice();

    void startRecording();
    void stopRecording();
    bool isRecording() const;

public slots:
    void saveRecording();

signals:
    void recordingStatusChanged(bool recording);

private slots:
    void handleDataReady();

private:
    void setupAudioFormat();
    void writeWavHeader(QFile &file, quint32 dataSize);
    void saveToWav();
    QVector<qint16> convertTo16kMono(const QVector<qint16> &input, int inRate, int inChannels);

    QAudioSource *audioSource = nullptr;
    QIODevice *audioIODevice = nullptr;
    QAudioFormat format;
    QQueue<QByteArray> audioBuffer;
    QByteArray tempBuffer;
    bool recording = false;
    const int targetSampleRate = 16000;
    const int targetChannelCount = 1;
    const int sampleSize = 16;
    int actualSampleRate = 16000;
    int actualChannelCount = 1;
};

#endif // SENSEVOICE_H
