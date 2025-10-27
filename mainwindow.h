#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include "audiocapture.h"
#include "sensevoice.h"
#include <QMainWindow>

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private:
    Ui::MainWindow *ui;

    SenseVoice *senseVoice;
    AudioCapture *audioCapture;

private slots:
    void toggleRecording();
    void updateRecordingStatus(bool recording);

};
#endif // MAINWINDOW_H
