import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout

# 建立application对象# Python学习群 872937351
app = QApplication(sys.argv)
# 建立窗体对象
w = QWidget()
# 设置窗体大小
w.resize(500, 500)

# 设置样式
w.layout = QVBoxLayout()
w.label = QLabel("Hello World!")
w.label.setStyleSheet("font-size:25px;margin-left:155px;")
w.setWindowTitle("PyQt5 窗口")
w.layout.addWidget(w.label)
w.setLayout(w.layout)

# 显示窗体
w.show()
# 运行程序
sys.exit(app.exec_())