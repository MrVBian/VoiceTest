#include "mainwindow.h"
#include "./ui_mainwindow.h"


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "sherpa-onnx/c-api/c-api.h"

#include <QDebug>
#include <QDir>
#include <QMessageBox>


MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    QString appDir = QCoreApplication::applicationDirPath();
    connect(ui->testBtn, &QPushButton::clicked, this, [this, appDir]() {
        const char *wav_filename =
            "sherpa-onnx-paraformer-zh-small/0.wav";
        const char *model_filename =
            "sherpa-onnx-paraformer-zh-small/model.int8.onnx";
        const char *tokens_filename =
            "sherpa-onnx-paraformer-zh-small/tokens.txt";
        const char *provider = "cpu";

        const SherpaOnnxWave *wave = SherpaOnnxReadWave(wav_filename);
        if (wave == NULL) {
            fprintf(stderr, "Failed to read %s\n", wav_filename);
            return ;
        }

        // Paraformer config
        SherpaOnnxOfflineParaformerModelConfig paraformer_config;
        memset(&paraformer_config, 0, sizeof(paraformer_config));
        paraformer_config.model = model_filename;

        // Offline model config
        SherpaOnnxOfflineModelConfig offline_model_config;
        memset(&offline_model_config, 0, sizeof(offline_model_config));
        offline_model_config.debug = 1;
        offline_model_config.num_threads = 1;
        offline_model_config.provider = provider;
        offline_model_config.tokens = tokens_filename;
        offline_model_config.paraformer = paraformer_config;

        // Recognizer config
        SherpaOnnxOfflineRecognizerConfig recognizer_config;
        memset(&recognizer_config, 0, sizeof(recognizer_config));
        recognizer_config.decoding_method = "greedy_search";
        recognizer_config.model_config = offline_model_config;

        const SherpaOnnxOfflineRecognizer *recognizer =
            SherpaOnnxCreateOfflineRecognizer(&recognizer_config);

        if (recognizer == NULL) {
            fprintf(stderr, "Please check your config!\n");
            SherpaOnnxFreeWave(wave);
            return;
        }

        const SherpaOnnxOfflineStream *stream =
            SherpaOnnxCreateOfflineStream(recognizer);

        SherpaOnnxAcceptWaveformOffline(stream, wave->sample_rate, wave->samples,
                                        wave->num_samples);
        SherpaOnnxDecodeOfflineStream(recognizer, stream);
        const SherpaOnnxOfflineRecognizerResult *result =
            SherpaOnnxGetOfflineStreamResult(stream);

        fprintf(stderr, "Decoded text: %s\n", result->text);

        SherpaOnnxDestroyOfflineRecognizerResult(result);
        SherpaOnnxDestroyOfflineStream(stream);
        SherpaOnnxDestroyOfflineRecognizer(recognizer);
        SherpaOnnxFreeWave(wave);
    });


    senseVoice = new SenseVoice(this);
    audioCapture = new AudioCapture(this);

    connect(ui->testBtn2, &QPushButton::clicked, this, [this, appDir]() {
        audioCapture->startCapture();
    });

    connect(ui->testBtn3, &QPushButton::clicked, this, [this, appDir]() {
        audioCapture->stopCapture();  // 停止录音
    });

    // connect(ui->testBtn2, &QPushButton::clicked, this, &MainWindow::on_testBtn2_clicked);
    // connect(senseVoice, &SenseVoice::recordingStatusChanged,
    //         this, &MainWindow::updateRecordingStatus);

    // connect(ui->testBtn2, &QPushButton::clicked, this, &MainWindow::toggleRecording);

    // connect(ui->testBtn3, &QPushButton::clicked,
    //         senseVoice, &SenseVoice::saveRecording);
}


void MainWindow::toggleRecording()
{
    if (senseVoice->isRecording()) {
        senseVoice->stopRecording();
    } else {
        senseVoice->startRecording();
    }
}

void MainWindow::updateRecordingStatus(bool recording)
{
    ui->testBtn2->setText(recording ? "Stop Recording" : "Start Recording");
    ui->label->setText(recording ? "Recording..." : "Ready");
}


MainWindow::~MainWindow()
{
    delete ui;
}
