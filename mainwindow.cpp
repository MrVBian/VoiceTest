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

    connect(ui->testBtn4, &QPushButton::clicked, this, [this]() {
        // 加载音频
        const char *wav_filename = "vad/lei-jun-test.wav";
        if (!SherpaOnnxFileExists(wav_filename)) {
            fprintf(stderr, "Please download %s\n", wav_filename);
        }
        const SherpaOnnxWave *wave = SherpaOnnxReadWave(wav_filename);
        if (wave == NULL) {
            fprintf(stderr, "Failed to read %s\n", wav_filename);
            return;
        }
        if (wave->sample_rate != 16000) {
            fprintf(stderr, "Expect the sample rate to be 16000. Given: %d\n",
                    wave->sample_rate);
            SherpaOnnxFreeWave(wave);
            return;
        }


        // 初始化模型
        const char *vad_filename;
        int32_t use_silero_vad = 0;
        int32_t use_ten_vad = 0;

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

        SherpaOnnxVadModelConfig vadConfig;
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

        const SherpaOnnxVoiceActivityDetector *vad =
            SherpaOnnxCreateVoiceActivityDetector(&vadConfig, 30);


        if (vad == NULL) {
            fprintf(stderr, "Please check your recognizer config!\n");
            SherpaOnnxFreeWave(wave);
            return ;
        }

        int32_t window_size = use_silero_vad ? vadConfig.silero_vad.window_size
                                             : vadConfig.ten_vad.window_size;

        int32_t i = 0;
        int is_eof = 0;

        while (!is_eof) {
            if (i + window_size < wave->num_samples) {
                SherpaOnnxVoiceActivityDetectorAcceptWaveform(vad, wave->samples + i,
                                                              window_size);
            } else {
                SherpaOnnxVoiceActivityDetectorFlush(vad);
                is_eof = 1;
            }
            while (!SherpaOnnxVoiceActivityDetectorEmpty(vad)) {
                const SherpaOnnxSpeechSegment *segment =
                    SherpaOnnxVoiceActivityDetectorFront(vad);

                float start = segment->start / 16000.0f;
                float duration = segment->n / 16000.0f;
                float stop = start + duration;

                fprintf(stderr, "%.3f -- %.3f\n", start, stop);
                // fprintf(stderr, "%.3f -- %.3f: %s\n", start, stop, result->text);

                SherpaOnnxDestroySpeechSegment(segment);
                SherpaOnnxVoiceActivityDetectorPop(vad);
            }
            i += window_size;
        }


        SherpaOnnxDestroyVoiceActivityDetector(vad);
        SherpaOnnxFreeWave(wave);
    });


    audioCapture = new AudioCapture(this);

    connect(ui->testBtn2, &QPushButton::clicked, this, [this, appDir]() {
        audioCapture->startCapture();
    });

    connect(ui->testBtn3, &QPushButton::clicked, this, [this, appDir]() {
        audioCapture->stopCapture();  // 停止录音
    });
}

MainWindow::~MainWindow()
{
    delete ui;
}
