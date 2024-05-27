# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'form.ui'
##
## Created by: Qt User Interface Compiler version 6.3.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox,
    QFormLayout, QFrame, QGridLayout, QGroupBox,
    QLabel, QLineEdit, QPushButton, QSizePolicy,
    QSlider, QSpacerItem, QSpinBox, QVBoxLayout,
    QWidget)

class Ui_V2Form(object):
    def setupUi(self, V2Form):
        if not V2Form.objectName():
            V2Form.setObjectName(u"V2Form")
        V2Form.resize(720, 620)
        self.gridLayout = QGridLayout(V2Form)
        self.gridLayout.setObjectName(u"gridLayout")
        self.frame_2 = QFrame(V2Form)
        self.frame_2.setObjectName(u"frame_2")
        sizePolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_2.sizePolicy().hasHeightForWidth())
        self.frame_2.setSizePolicy(sizePolicy)
        self.frame_2.setMinimumSize(QSize(255, 602))
        self.gridLayout_7 = QGridLayout(self.frame_2)
        self.gridLayout_7.setObjectName(u"gridLayout_7")
        self.gridLayout_7.setContentsMargins(-1, -1, -1, 0)
        self.groupBox_2 = QGroupBox(self.frame_2)
        self.groupBox_2.setObjectName(u"groupBox_2")
        sizePolicy1 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.groupBox_2.sizePolicy().hasHeightForWidth())
        self.groupBox_2.setSizePolicy(sizePolicy1)
        self.formLayout = QFormLayout(self.groupBox_2)
        self.formLayout.setObjectName(u"formLayout")
        self.deviceComboBox = QComboBox(self.groupBox_2)
        self.deviceComboBox.setObjectName(u"deviceComboBox")
        sizePolicy1.setHeightForWidth(self.deviceComboBox.sizePolicy().hasHeightForWidth())
        self.deviceComboBox.setSizePolicy(sizePolicy1)

        self.formLayout.setWidget(0, QFormLayout.SpanningRole, self.deviceComboBox)

        self.undistortCheckBox = QCheckBox(self.groupBox_2)
        self.undistortCheckBox.setObjectName(u"undistortCheckBox")
        sizePolicy2 = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.undistortCheckBox.sizePolicy().hasHeightForWidth())
        self.undistortCheckBox.setSizePolicy(sizePolicy2)
        self.undistortCheckBox.setMinimumSize(QSize(72, 0))
        self.undistortCheckBox.setChecked(True)

        self.formLayout.setWidget(1, QFormLayout.LabelRole, self.undistortCheckBox)

        self.liveFeedCheckBox = QCheckBox(self.groupBox_2)
        self.liveFeedCheckBox.setObjectName(u"liveFeedCheckBox")
        sizePolicy2.setHeightForWidth(self.liveFeedCheckBox.sizePolicy().hasHeightForWidth())
        self.liveFeedCheckBox.setSizePolicy(sizePolicy2)
        self.liveFeedCheckBox.setMinimumSize(QSize(72, 0))
        self.liveFeedCheckBox.setChecked(True)

        self.formLayout.setWidget(1, QFormLayout.FieldRole, self.liveFeedCheckBox)

        self.label_4 = QLabel(self.groupBox_2)
        self.label_4.setObjectName(u"label_4")
        sizePolicy2.setHeightForWidth(self.label_4.sizePolicy().hasHeightForWidth())
        self.label_4.setSizePolicy(sizePolicy2)
        self.label_4.setMinimumSize(QSize(72, 0))

        self.formLayout.setWidget(2, QFormLayout.LabelRole, self.label_4)

        self.exposureSlider = QSlider(self.groupBox_2)
        self.exposureSlider.setObjectName(u"exposureSlider")
        self.exposureSlider.setEnabled(True)
        sizePolicy1.setHeightForWidth(self.exposureSlider.sizePolicy().hasHeightForWidth())
        self.exposureSlider.setSizePolicy(sizePolicy1)
        self.exposureSlider.setMinimum(200)
        self.exposureSlider.setMaximum(16000)
        self.exposureSlider.setValue(10000)
        self.exposureSlider.setOrientation(Qt.Horizontal)

        self.formLayout.setWidget(2, QFormLayout.FieldRole, self.exposureSlider)


        self.gridLayout_7.addWidget(self.groupBox_2, 0, 0, 1, 1)

        self.groupBox_3 = QGroupBox(self.frame_2)
        self.groupBox_3.setObjectName(u"groupBox_3")
        sizePolicy1.setHeightForWidth(self.groupBox_3.sizePolicy().hasHeightForWidth())
        self.groupBox_3.setSizePolicy(sizePolicy1)
        self.gridLayout_3 = QGridLayout(self.groupBox_3)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.widthBitsPushButton = QPushButton(self.groupBox_3)
        self.widthBitsPushButton.setObjectName(u"widthBitsPushButton")
        sizePolicy1.setHeightForWidth(self.widthBitsPushButton.sizePolicy().hasHeightForWidth())
        self.widthBitsPushButton.setSizePolicy(sizePolicy1)

        self.gridLayout_3.addWidget(self.widthBitsPushButton, 0, 0, 1, 2)

        self.heightBitsPushButton = QPushButton(self.groupBox_3)
        self.heightBitsPushButton.setObjectName(u"heightBitsPushButton")
        sizePolicy1.setHeightForWidth(self.heightBitsPushButton.sizePolicy().hasHeightForWidth())
        self.heightBitsPushButton.setSizePolicy(sizePolicy1)

        self.gridLayout_3.addWidget(self.heightBitsPushButton, 1, 0, 1, 2)


        self.gridLayout_7.addWidget(self.groupBox_3, 3, 0, 1, 1)

        self.savePushButton = QPushButton(self.frame_2)
        self.savePushButton.setObjectName(u"savePushButton")
        self.savePushButton.setEnabled(False)
        sizePolicy1.setHeightForWidth(self.savePushButton.sizePolicy().hasHeightForWidth())
        self.savePushButton.setSizePolicy(sizePolicy1)

        self.gridLayout_7.addWidget(self.savePushButton, 7, 0, 1, 1)

        self.groupBox_5 = QGroupBox(self.frame_2)
        self.groupBox_5.setObjectName(u"groupBox_5")
        sizePolicy1.setHeightForWidth(self.groupBox_5.sizePolicy().hasHeightForWidth())
        self.groupBox_5.setSizePolicy(sizePolicy1)
        self.gridLayout_5 = QGridLayout(self.groupBox_5)
        self.gridLayout_5.setObjectName(u"gridLayout_5")
        self.label_12 = QLabel(self.groupBox_5)
        self.label_12.setObjectName(u"label_12")
        sizePolicy2.setHeightForWidth(self.label_12.sizePolicy().hasHeightForWidth())
        self.label_12.setSizePolicy(sizePolicy2)
        self.label_12.setMinimumSize(QSize(72, 0))

        self.gridLayout_5.addWidget(self.label_12, 0, 0, 1, 1)

        self.displayDelaySpinBox = QDoubleSpinBox(self.groupBox_5)
        self.displayDelaySpinBox.setObjectName(u"displayDelaySpinBox")
        self.displayDelaySpinBox.setEnabled(True)
        sizePolicy1.setHeightForWidth(self.displayDelaySpinBox.sizePolicy().hasHeightForWidth())
        self.displayDelaySpinBox.setSizePolicy(sizePolicy1)
        self.displayDelaySpinBox.setMinimum(20.000000000000000)
        self.displayDelaySpinBox.setMaximum(1000.000000000000000)
        self.displayDelaySpinBox.setSingleStep(10.000000000000000)
        self.displayDelaySpinBox.setValue(100.000000000000000)

        self.gridLayout_5.addWidget(self.displayDelaySpinBox, 1, 1, 1, 1)

        self.label_13 = QLabel(self.groupBox_5)
        self.label_13.setObjectName(u"label_13")
        sizePolicy2.setHeightForWidth(self.label_13.sizePolicy().hasHeightForWidth())
        self.label_13.setSizePolicy(sizePolicy2)
        self.label_13.setMinimumSize(QSize(72, 0))

        self.gridLayout_5.addWidget(self.label_13, 1, 0, 1, 1)

        self.displayIndexSpinBox = QSpinBox(self.groupBox_5)
        self.displayIndexSpinBox.setObjectName(u"displayIndexSpinBox")
        sizePolicy1.setHeightForWidth(self.displayIndexSpinBox.sizePolicy().hasHeightForWidth())
        self.displayIndexSpinBox.setSizePolicy(sizePolicy1)

        self.gridLayout_5.addWidget(self.displayIndexSpinBox, 0, 1, 1, 1)


        self.gridLayout_7.addWidget(self.groupBox_5, 1, 0, 1, 1)

        self.groupBox_4 = QGroupBox(self.frame_2)
        self.groupBox_4.setObjectName(u"groupBox_4")
        sizePolicy1.setHeightForWidth(self.groupBox_4.sizePolicy().hasHeightForWidth())
        self.groupBox_4.setSizePolicy(sizePolicy1)
        self.gridLayout_4 = QGridLayout(self.groupBox_4)
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.polyFitPushButton = QPushButton(self.groupBox_4)
        self.polyFitPushButton.setObjectName(u"polyFitPushButton")

        self.gridLayout_4.addWidget(self.polyFitPushButton, 0, 0, 1, 2)

        self.label_8 = QLabel(self.groupBox_4)
        self.label_8.setObjectName(u"label_8")
        sizePolicy2.setHeightForWidth(self.label_8.sizePolicy().hasHeightForWidth())
        self.label_8.setSizePolicy(sizePolicy2)
        self.label_8.setMinimumSize(QSize(72, 0))

        self.gridLayout_4.addWidget(self.label_8, 2, 0, 1, 1)

        self.fileNameLineEdit = QLineEdit(self.groupBox_4)
        self.fileNameLineEdit.setObjectName(u"fileNameLineEdit")
        sizePolicy1.setHeightForWidth(self.fileNameLineEdit.sizePolicy().hasHeightForWidth())
        self.fileNameLineEdit.setSizePolicy(sizePolicy1)

        self.gridLayout_4.addWidget(self.fileNameLineEdit, 2, 1, 1, 1)

        self.validatePushButton = QPushButton(self.groupBox_4)
        self.validatePushButton.setObjectName(u"validatePushButton")

        self.gridLayout_4.addWidget(self.validatePushButton, 1, 0, 1, 2)


        self.gridLayout_7.addWidget(self.groupBox_4, 4, 0, 1, 1)

        self.groupBox = QGroupBox(self.frame_2)
        self.groupBox.setObjectName(u"groupBox")
        sizePolicy1.setHeightForWidth(self.groupBox.sizePolicy().hasHeightForWidth())
        self.groupBox.setSizePolicy(sizePolicy1)
        self.gridLayout_2 = QGridLayout(self.groupBox)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.maskThresholdSlider = QSlider(self.groupBox)
        self.maskThresholdSlider.setObjectName(u"maskThresholdSlider")
        sizePolicy1.setHeightForWidth(self.maskThresholdSlider.sizePolicy().hasHeightForWidth())
        self.maskThresholdSlider.setSizePolicy(sizePolicy1)
        self.maskThresholdSlider.setMaximum(255)
        self.maskThresholdSlider.setValue(100)
        self.maskThresholdSlider.setOrientation(Qt.Horizontal)

        self.gridLayout_2.addWidget(self.maskThresholdSlider, 0, 1, 1, 1)

        self.createMaskPushButton = QPushButton(self.groupBox)
        self.createMaskPushButton.setObjectName(u"createMaskPushButton")
        sizePolicy1.setHeightForWidth(self.createMaskPushButton.sizePolicy().hasHeightForWidth())
        self.createMaskPushButton.setSizePolicy(sizePolicy1)

        self.gridLayout_2.addWidget(self.createMaskPushButton, 1, 0, 1, 2)

        self.label_6 = QLabel(self.groupBox)
        self.label_6.setObjectName(u"label_6")
        sizePolicy2.setHeightForWidth(self.label_6.sizePolicy().hasHeightForWidth())
        self.label_6.setSizePolicy(sizePolicy2)
        self.label_6.setMinimumSize(QSize(72, 0))

        self.gridLayout_2.addWidget(self.label_6, 0, 0, 1, 1)


        self.gridLayout_7.addWidget(self.groupBox, 2, 0, 1, 1)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout_7.addItem(self.verticalSpacer, 6, 0, 1, 1)


        self.gridLayout.addWidget(self.frame_2, 0, 0, 1, 1)

        self.frame = QFrame(V2Form)
        self.frame.setObjectName(u"frame")
        sizePolicy3 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.frame.sizePolicy().hasHeightForWidth())
        self.frame.setSizePolicy(sizePolicy3)
        self.frame.setFrameShape(QFrame.StyledPanel)
        self.frame.setFrameShadow(QFrame.Raised)
        self.verticalLayout_3 = QVBoxLayout(self.frame)
        self.verticalLayout_3.setSpacing(6)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.cameraFeedLabel = QLabel(self.frame)
        self.cameraFeedLabel.setObjectName(u"cameraFeedLabel")
        sizePolicy4 = QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        sizePolicy4.setHorizontalStretch(0)
        sizePolicy4.setVerticalStretch(1)
        sizePolicy4.setHeightForWidth(self.cameraFeedLabel.sizePolicy().hasHeightForWidth())
        self.cameraFeedLabel.setSizePolicy(sizePolicy4)
        self.cameraFeedLabel.setMinimumSize(QSize(400, 200))
        self.cameraFeedLabel.setPixmap(QPixmap(u"imgs/blank.png"))
        self.cameraFeedLabel.setScaledContents(True)

        self.verticalLayout_3.addWidget(self.cameraFeedLabel)

        self.resultAreaLabel = QLabel(self.frame)
        self.resultAreaLabel.setObjectName(u"resultAreaLabel")
        sizePolicy4.setHeightForWidth(self.resultAreaLabel.sizePolicy().hasHeightForWidth())
        self.resultAreaLabel.setSizePolicy(sizePolicy4)
        self.resultAreaLabel.setMinimumSize(QSize(400, 200))
        self.resultAreaLabel.setPixmap(QPixmap(u"imgs/blank.png"))
        self.resultAreaLabel.setScaledContents(True)

        self.verticalLayout_3.addWidget(self.resultAreaLabel)


        self.gridLayout.addWidget(self.frame, 0, 1, 1, 1)


        self.retranslateUi(V2Form)

        QMetaObject.connectSlotsByName(V2Form)
    # setupUi

    def retranslateUi(self, V2Form):
        V2Form.setWindowTitle(QCoreApplication.translate("V2Form", u"V2 Calibrator", None))
        self.groupBox_2.setTitle(QCoreApplication.translate("V2Form", u"Device", None))
        self.undistortCheckBox.setText(QCoreApplication.translate("V2Form", u"Undistort", None))
        self.liveFeedCheckBox.setText(QCoreApplication.translate("V2Form", u"Live feed", None))
        self.label_4.setText(QCoreApplication.translate("V2Form", u"Exposure", None))
        self.groupBox_3.setTitle(QCoreApplication.translate("V2Form", u"Gradients", None))
        self.widthBitsPushButton.setText(QCoreApplication.translate("V2Form", u"Measure width bits", None))
        self.heightBitsPushButton.setText(QCoreApplication.translate("V2Form", u"Measure height bits", None))
        self.savePushButton.setText(QCoreApplication.translate("V2Form", u"Save", None))
        self.groupBox_5.setTitle(QCoreApplication.translate("V2Form", u"Display", None))
        self.label_12.setText(QCoreApplication.translate("V2Form", u"Index", None))
        self.label_13.setText(QCoreApplication.translate("V2Form", u"Delay", None))
        self.groupBox_4.setTitle(QCoreApplication.translate("V2Form", u"Calibration", None))
        self.polyFitPushButton.setText(QCoreApplication.translate("V2Form", u"Fit a 3D polynomial", None))
        self.label_8.setText(QCoreApplication.translate("V2Form", u"File name", None))
        self.fileNameLineEdit.setText(QCoreApplication.translate("V2Form", u"out\\V2Out.json", None))
        self.validatePushButton.setText(QCoreApplication.translate("V2Form", u"Validate", None))
        self.groupBox.setTitle(QCoreApplication.translate("V2Form", u"Mask", None))
        self.createMaskPushButton.setText(QCoreApplication.translate("V2Form", u"Create mask", None))
        self.label_6.setText(QCoreApplication.translate("V2Form", u"Threshold", None))
        self.cameraFeedLabel.setText("")
        self.resultAreaLabel.setText("")
    # retranslateUi

