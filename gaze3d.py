# -*- coding: utf-8 -*-
"""
POLS 視野等表示 ver.9
Stand-alone

"""
# Import and setting ===========================================================================================
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.preprocessing import StandardScaler
import pickle
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import os

fp = FontProperties(fname="c:\\Windows\\Fonts\\YuGothM.ttc")#日本語フォント位置指定

deffont=('Yu Gothic', 20)
file_name = ""


class ShowViewPointapp(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        tk.Tk.iconbitmap(self, default="k-lab_logo.ico")
        tk.Tk.wm_title(self, "EOG Gaze Track 3D")

        container = ttk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        self.frames = {}

        for F in (StartPage, Page3D, PageOne):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()


class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        global sf
        style = ttk.Style()
        style.configure('TButton', font=deffont)
        ttk.Frame.__init__(self, parent)

        label = ttk.Label(self, text="CSVファイルを選択してください", font=deffont)
        label.pack(pady=10)

        button0 = ttk.Button(self, text='ファイル選択', style='TButton', command=lambda: self.load_file())
        button0.pack(pady=10, ipadx=10)

        button1 = ttk.Button(self, text="ファイルパスリセット", style='TButton', command=(lambda: self.cancel()))
        button1.pack(pady=10, ipadx=10)

        label2 = ttk.Label(self, text="サンプリング間隔", font=deffont)
        label2.pack(pady=10)

        sf = ttk.Spinbox(self, from_=1, to=20, width=5, increment=1, font=deffont)
        sf.pack(pady=10)

        label3 = ttk.Label(self, text="(msec)", font=deffont)
        label3.pack(pady=10)

        button_chk = ttk.Button(self, text="確定", command=(lambda: [self.check_encoding()]))
        button_chk.pack(pady=10, ipadx=10)

        button2 = ttk.Button(self, text="視野解析", command=(lambda: [self.Data_analysis(), controller.show_frame(PageOne)]))
        button2.pack(pady=10, ipadx=10)

        button4 = ttk.Button(self, text="3D", command=lambda: controller.show_frame(Page3D))
        button4.pack(pady=10, ipadx=10)

    def load_file(self):
        global file_name, label2

        initdir = os.path.dirname(__file__)        # スクリプト自身のファイルパスのディレクトリの取得
        # 初期ディレクトリとして自身の親ディレクトリを指定する
        file_name = filedialog.askopenfilename(filetypes=[("CSV Files", ".csv")],initialdir=initdir)
        if file_name != "":
            label2 = ttk.Label(self, text="%s" % file_name, font=deffont)
            label2.pack(pady=10)

    def cancel(self):
        global file_name, label2

        if file_name != "":
            label2.destroy()
            file_name = ""

    def sampling_frequency(self):
        sf_values = int(sf.get())
        freq=1000/sf_values
        return freq



    def check_encoding(self):

        if file_name == "":
            sub_win = tk.Toplevel()
            tk.Message(sub_win, aspect=1000, text="CSVファイルを選択してください",
                       font=deffont).pack(pady=10, padx=10)
        else:
            from chardet.universaldetector import UniversalDetector
            """
            pathnameから該当するファイルの文字コードを判別して
            ファイル名と文字コードのdictを返す
    
            :param pathname: 文字コードを判別したいフォルダ
            :return: ファイル名がキー、文字コードが値のdict
            """
            detector = UniversalDetector()
            with open(file_name, mode='rb') as f:
                for binary in f:
                    detector.feed(binary)
                    if detector.done:
                        break
            detector.close()
            # print(detector.result, end='')
            print(detector.result['encoding'], end='')





    def Data_analysis(self):
        global file_name,pos, move_point,chkidx,xylim,AnalysisData,signal

        if file_name == "":
            sub_win = tk.Toplevel()
            tk.Message(sub_win, aspect=1000, text="CSVファイルを選択してください",
                       font=deffont).pack(pady=10, padx=10)


        # Pols data analysis =======================================================================================
        # signal = np.loadtxt("%s" % file_name, skiprows=1, delimiter=",", encoding="utf-8")
        # signal = np.loadtxt("%s" % file_name, skiprows=1, delimiter=",")
        # signal = np.loadtxt("%s" % file_name, skiprows=1, delimiter=",",encoding="SHIFT_JIS")

        # 　csv ファイルを行列式として読み込む（utf-8形式）




        """
        Memo -------------------------------------------------------------------------------------------------
        signal:
        0=No,1=Idx_Raw,2=テーブル番号,3=絶対位置X,4=絶対位置Y,5=表示状態,6=状態連番,
        7=信号Ch0(水平信号(左眼反転)),8=信号(Ch1垂直信号(mV)),9=信号Ch2(水平眼位(左眼反転)),10=信号Ch3(垂直眼位)
        ------------------------------------------------------------------------------------------------------
        """
        (t, s) = signal.shape  # 行列数表示: t=行数(時間データ),s=列数(計測項目データ)
        chkidx = np.where((signal[:, 6] == 0) & (signal[:, 5] == 1))[0]  # 検査開始時間の抽出
        r = chkidx.size  # 検査数を計測
        chkidx = np.hstack([0, chkidx])  # chkidx:検査開始時間一覧　(原点を含めた検査開始時間一覧の作成)
        chkidx = chkidx.astype(int)  # int変換
        d_real = np.diff(signal[chkidx, 3:5], axis=0)  # X軸Y軸それぞれの絶対座標に対して差分をとることで相対座標を作成
        d_real = np.vstack([[0, 0], d_real])  # 原点を含めた相対座標一覧の作成
        samplingfreq = self.sampling_frequency()
        starttime = chkidx / samplingfreq  # 検査開始時間(sec)

        # FFT
        fft_sig = np.fft.fft(signal[:, 7:9], axis=0)  # X軸Y軸それぞれに対してFFT
        freq = np.fft.fftfreq(t, d=(1 / samplingfreq))  # FFTの周波数表示
        # Band pass filter(0.02hz以上9.8hz以下の周波数のみ通す)
        bandpassidx = np.where((np.abs(freq) < 0.02) | (np.abs(freq) > 9.8))[0]
        fft_sig[bandpassidx, :] = 0

        ifft_sig = np.real(np.fft.ifft(fft_sig, axis=0))  # X軸Y軸それぞれに対して逆フーリエ変換

        pos = np.cumsum(ifft_sig, axis=0)  # 位置座標一覧の作成(逆フーリエ変換した信号をX軸Y軸それぞれに対して積分)
        acc = np.diff(ifft_sig, axis=0)  # X軸Y軸それぞれの変化量に対して差分をとることで加速度の作成
        acc = np.vstack([[0, 0], acc])  # 原点を含めた加速度一覧の作成)

        # Prediction Move point ================================================================================================

        """
        SVM(サポートベクトルマシン)を用いて視線の動きを判定    
        判定結果：　0=視線が動いていない(固視), 1=視線が動いている(視線移動),  2=瞬き(瞬目)    
        学習済み svm model path : "C:\\Pols_seki\\pols_SVM_model.sav"
        """
        # Load SVM model
        # filename = "pols_SVM_model.sav"
        filename = "C:\\Python\\Pols_disp\\pols_SVM_model.sav"
        model = pickle.load(open(filename, 'rb'))  # Load SVM model
        print("Successfully Load SVM-model")

        sigx = np.hstack((ifft_sig, acc))  # 判定に使う行列作成
        print("ifft+ACC")

        # 数値を正規化
        sc = StandardScaler()
        sc.fit(sigx)
        x = sc.transform(sigx)

        prediction = model.predict(x)  # svm Prediction

        # Calc TimeData---------------------------------------------------------------------
        AnalysisData = np.zeros((chkidx.size, 11))  # 分析後のデータ格納
        """
        AnalysisData : 
        [Xdeg(Real) ,Ydeg(Real) ,RT ,ToF ,Fix2time ,Blink ,RT_decision ,ToF_decision, Xdeg(Abs) ,Ydeg(Abs),times(sec)]
        """
        AnalysisData[:, 0:2] = d_real  # 相対座標
        AnalysisData[:, 8:10] = signal[chkidx, 3:5]  # 絶対座標
        AnalysisData[:, 10] = starttime.T  # 検査開始時間(sec)

        move_point = np.zeros(t)
        maxmin = np.zeros((r + 1, 2))

        for ck in range(0, t - 4):
            movechk = np.where(prediction[ck:ck + 5] == 1)[0]
            blinkchk = np.where(prediction[ck:ck + 5] == 2)[0]
            if movechk.size > 2:
                move_point[ck:ck + 5] = 1
            if blinkchk.size > 2:
                move_point[ck:ck + 5] = 2

        for cn in range(0, r + 1):
            if cn == r:
                caldata = pos[chkidx[cn]:t, :]
                pre_cn = move_point[chkidx[cn]:t]
            else:
                caldata = pos[chkidx[cn]:chkidx[cn + 1], :]
                pre_cn = move_point[chkidx[cn]:chkidx[cn + 1]]

            maxmin[cn, :] = np.amax(caldata, axis=0) - np.amin(caldata, axis=0)

            findmove = np.where(pre_cn == 1)[0]
            findfix = np.where(pre_cn == 0)[0]
            findblink = np.where(pre_cn == 2)[0]

            chk_fixpoint = np.zeros(1)

            if findfix.size > 0:
                chk_fixpoint = findfix[0]
                for fn in range(findfix.size - 1):
                    if findfix[fn + 1] != findfix[fn] + 1:
                        chk_fixpoint = np.hstack((chk_fixpoint, findfix[fn:fn + 2]))

                chk_fixpoint = np.hstack((chk_fixpoint, findfix[- 1]))
            else:
                chk_fixpoint = np.hstack((chk_fixpoint, findmove[0]))

            chksize = chk_fixpoint.size
            reactiontime = (chk_fixpoint[1] - chk_fixpoint[0]) / samplingfreq
            timeofflight = findmove.size / samplingfreq
            if chk_fixpoint.size > 3:
                FT2_fix = (findfix.size / samplingfreq) - reactiontime
            else:
                FT2_fix = 0

            if findblink.size != 0:
                if findblink.size <= 0.1 * samplingfreq:
                    Blink_time = 0
                    move_point[findblink] = 1
                else:
                    Blink_time = (findblink.size) / samplingfreq

                AnalysisData[cn, 5] = Blink_time

            if findblink.size != 0:
                chk_bt = findblink[0]
                for bt in range(findblink.size - 1):
                    if findblink[bt + 1] != findblink[bt] + 1:
                        chk_bt = np.hstack((chk_bt, findblink[bt:bt + 2]))

                chk_bt = np.hstack((chk_bt, findblink[-1]))
            AnalysisData[cn, 2] = np.round(reactiontime, decimals=3)
            AnalysisData[cn, 3] = np.round(timeofflight, decimals=3)
            AnalysisData[cn, 4] = np.round(FT2_fix, decimals=2)

            if reactiontime > 0.7:
                AnalysisData[cn, 6] = 1
            if timeofflight > 0.6:
                AnalysisData[cn, 7] = 1

        xylim = np.amax(maxmin)


class Page3D(tk.Frame):

    def __init__(self, parent, controller):
        global s1, s2, sf
        ttk.Frame.__init__(self, parent)

        label = ttk.Label(self, text="表示する検査点を選択してください", font=deffont)
        label.grid(pady=10, padx=10, row=0, column=0, columnspan=3)

        s1 = ttk.Spinbox(self, from_=0, to=77, increment=1, width=10, font=deffont)
        s1.grid(pady=10, padx=10, row=1, column=0)

        label1 = ttk.Label(self, text="~", font=deffont)
        label1.grid(row=1, column=1)

        s2 = ttk.Spinbox(self, from_=1, to=77, increment=1, width=10, font=deffont)
        s2.grid(pady=10, padx=10, row=1, column=2)

        button1 = ttk.Button(self, text='表示', style='TButton', command=lambda: self.gaze_3d())
        button1.grid(pady=10, padx=10, row=3, column=0, columnspan=3)

        button2 = ttk.Button(self, text="戻る", command=lambda: controller.show_frame(StartPage))
        button2.grid(pady=10, padx=10, row=4, column=0, columnspan=3)

    def spin_values(self):
        s1_values = int(s1.get())
        s2_values = int(s2.get())
        return s1_values, s2_values

    def gaze_3d(self):

        if file_name == "":
            sub_win = tk.Toplevel()
            tk.Message(sub_win, aspect=1000, text="CSVファイルを選択してください",
                       font=deffont).pack(pady=10, padx=10)

        else:
            signal = np.loadtxt("%s" % file_name, skiprows=1, delimiter=",", encoding="utf-8")
            # signal = np.loadtxt("%s" %file_name, skiprows=1, delimiter=",")
            # 　csv ファイルを行列式として読み込む（utf-8形式）

            """
            Memo -------------------------------------------------------------------------------------------------
            signal:
            0=No,1=Idx_Raw,2=テーブル番号,3=絶対位置X,4=絶対位置Y,5=表示状態,6=状態連番,
            7=信号Ch0(水平信号(左眼反転)),8=信号(Ch1垂直信号(mV)),9=信号Ch2(水平眼位(左眼反転)),10=信号Ch3(垂直眼位)
            ------------------------------------------------------------------------------------------------------
            """
            (t, s) = signal.shape  # 行列数表示: t=行数(時間データ),s=列数(計測項目データ)
            chkidx = np.where((signal[:, 6] == 0) & (signal[:, 5] == 1))[0]  # 検査開始時間の抽出
            r = chkidx.size  # 検査数を計測
            chkidx = np.hstack([0, chkidx])  # chkidx:検査開始時間一覧　(原点を含めた検査開始時間一覧の作成)
            chkidx = chkidx.astype(int)  # int変換
            d_real = np.diff(signal[chkidx, 3:5], axis=0)  # X軸Y軸それぞれの絶対座標に対して差分をとることで相対座標を作成
            d_real = np.vstack([[0, 0], d_real])  # 原点を含めた相対座標一覧の作成

            samplingfreq =StartPage.sampling_frequency(self)
            stime = np.arange(t) / samplingfreq  # 検査時間(sec)

            # FFT --------------------------------------------------------------------
            fft_sig = np.fft.fft(signal[:, 7:9], axis=0)  # X軸Y軸それぞれに対してFFT
            freq = np.fft.fftfreq(t, d=(1 / samplingfreq))
            bandpassidx = np.where((np.abs(freq) < 0.02) | (np.abs(freq) > 9.8))[0]
            fft_sig[bandpassidx, :] = 0
            ifft_sig = np.real(np.fft.ifft(fft_sig, axis=0))
            pos = np.cumsum(ifft_sig, axis=0)  # 位置座標一覧の作成(逆フーリエ変換した信号をX軸Y軸それぞれに対して積分)
            s1, s2 = self.spin_values()

            startidx = chkidx[s1]
            if s2 >= r:
                endidx = t
                print(endidx)
            elif s1 > s2:
                startidx = chkidx[s2 + 1]
                endidx = chkidx[s1]
            else:
                startidx = chkidx[s1]
                endidx = chkidx[s2 + 1]

            fig = plt.figure(figsize=(10, 9))
            ax = Axes3D(fig)
            ax.plot(stime[startidx:endidx], pos[startidx:endidx, 0], pos[startidx:endidx, 1], ".-",
                    label="Eye Movement")
            ax.set_xlabel("Time(sec)")
            ax.set_ylabel("Horizontal Signal")
            ax.set_zlabel("Vertical Signal")
            plt.grid()
            plt.show()


class PageOne(tk.Frame):
    # Setting Button-------------------------------------------------------------------------------------
    def __init__(self, parent, controller):
        style = ttk.Style()
        style.configure('TButton', font=deffont)
        tk.Frame.__init__(self, parent)

        label1 = tk.Label(self, text="表示したいデータを選択してください", font=deffont)
        label1.pack(pady=10, padx=30)

        button_vf = ttk.Button(self, text="視野データ", command=lambda: self.VisualFieldPlot())
        button_vf.pack(pady=5, padx=10, fill=tk.X)

        button_rtof = ttk.Button(self, text="時間データ", command=lambda: self.RtandTofPlot())
        button_rtof.pack(pady=5, padx=10, fill=tk.X)

        button_all = ttk.Button(self, text="EOGデータ", command=lambda: self.TimedataPlot())
        button_all.pack(pady=5, padx=10, fill=tk.X)

        button_return = ttk.Button(self, text="戻る", command=lambda: controller.show_frame(StartPage))
        button_return.pack(pady=5, padx=10, fill=tk.X)

    # VisualField-------------------------------------------------------------------------------------
    def VisualFieldPlot(self):

        mc2 = move_point
        deltamax = xylim

        if "L.csv" in file_name:
            LR = 0
        else:
            LR = 1

        FalseRT = np.where((AnalysisData[1:-1, 6] == 1))[0] + 1
        SlowTOF = np.where((AnalysisData[1:-1, 7] == 1))[0] + 1
        Blink = np.where((AnalysisData[1:-1, 5] > 0.18))[0] + 1

        # click marker-------------------------------------------------------------------------
        def onclick(event):
            click_x = round(event.xdata)
            click_y = round(event.ydata)
            pos_cn = np.where((AnalysisData[:, 0] == click_x) & (AnalysisData[:, 1] == click_y))[0]

            if pos_cn.size == 0:
                pos_cn = np.where((AnalysisData[:, 0] <= click_x + 1) & (AnalysisData[:, 0] >= click_x - 1)
                                  & (AnalysisData[:, 1] <= click_y + 1) & (AnalysisData[:, 1] >= click_y - 1))[0]

            if pos_cn.size != 0:
                # Check False point-----------------------------------
                pos_fnum = 0
                pos_snum = 0
                pos_bnum = 0

                for i in range(pos_cn.size):
                    chk_fp = np.where(FalseRT == pos_cn[i])[0]
                    if chk_fp.size != 0:
                        pos_fnum = FalseRT[chk_fp]

                for j in range(pos_cn.size):
                    chk_sp = np.where(SlowTOF == pos_cn[j])[0]
                    if chk_sp.size != 0:
                        pos_snum = SlowTOF[chk_sp]

                for k in range(pos_cn.size):
                    chk_bp = np.where(Blink == pos_cn[k])[0]
                    if chk_bp.size != 0:
                        pos_bnum = Blink[chk_bp]

                if pos_fnum != 0:
                    pos_num = pos_fnum
                elif pos_snum != 0:
                    pos_num = pos_snum
                elif pos_bnum != 0:
                    pos_num = pos_bnum
                else:
                    pos_num = pos_cn[-1]

                pos_cn = int(pos_num)

                if pos_cn == chkidx.size - 1:
                    cal_xy = pos[chkidx[pos_cn]:-1, ]
                    cal_mc2 = mc2[chkidx[pos_cn]:-1, ]
                else:
                    cal_xy = pos[chkidx[pos_cn]:chkidx[pos_cn + 1], ]
                    cal_mc2 = mc2[chkidx[pos_cn]:chkidx[pos_cn + 1], ]

                # find eye move
                findmove = np.where(cal_mc2 > 0)[0]
                findfix = np.where(cal_mc2 == 0)[0]
                if pos_bnum != 0:
                    findblink = np.where(cal_mc2 == 2)[0]

                if findfix.size > 0:
                    fix_old = findfix.size - 1
                    fix_new = findfix[fix_old]
                    for fn in range(findfix.size - 1):
                        if findfix[fn + 1] != findfix[fn] + 1:
                            fix_old = findfix[fn] + 1
                            fix_new = findfix[fn + 1:-1]
                            break

                plt.style.use("seaborn")
                if LR < 1:
                    gp2d = fig.add_subplot(122)
                else:
                    gp2d = fig.add_subplot(121)

                gp2d.cla()
                # Fix max min ----------------------------------------------
                meanxy = np.mean(cal_xy, axis=0)
                maxmax = np.amax(deltamax)
                xmin = meanxy[0] - maxmax
                xmax = meanxy[0] + maxmax
                ymin = meanxy[1] - maxmax
                ymax = meanxy[1] + maxmax

                """
                # movie-----------------------------------------------------------
                plt.title("%s \n (%s -> %s)" % (AnalysisData[pos_cn, 0:2], AnalysisData[pos_cn-1, 8:10], AnalysisData[pos_cn, 8:10]), fontdict={"fontsize": 13})
                plotsize=chkidx[pos_cn + 1] - chkidx[pos_cn] + 1
                for num in range(plotsize):
                    plt.cla()
                    gp2d.plot(cal_xy[0:num,0], cal_xy[0:num,1], color="deepskyblue")
                    gp2d.plot(cal_xy[num-1,0], cal_xy[num-1,1], "o", color="deepskyblue")
                    plt.xlim(xmin, xmax)
                    plt.ylim(ymin, ymax)
                    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
                    plt.pause(0.01)
                """

                # plot-----------------------------------------------------------
                if pos_cn != 0:
                    plt.title("No.%s %s \n (%s -> %s)" % (pos_cn,
                    AnalysisData[pos_cn, 0:2], AnalysisData[pos_cn - 1, 8:10], AnalysisData[pos_cn, 8:10]),
                              fontdict={"fontsize": 13})
                    gp2d.plot(cal_xy[:, 0], cal_xy[:, 1], "-", color="deepskyblue")
                    gp2d.plot(cal_xy[findfix[0]:fix_old, 0], cal_xy[findfix[0]:fix_old, 1], "o", color="gold",
                              label="Reaction Time(RT)%s(sec)" % (AnalysisData[pos_cn, 2]))
                    gp2d.plot(cal_xy[findmove, 0], cal_xy[findmove, 1], ".", color="deepskyblue",
                              label="Time of Flight(ToF)%s(sec)" % (AnalysisData[pos_cn, 3]))
                    if pos_bnum != 0:
                        gp2d.plot(cal_xy[findblink, 0], cal_xy[findblink, 1], "b*", label="Blink")

                    gp2d.plot(cal_xy[fix_new, 0], cal_xy[fix_new, 1], "ro",
                              label="fixation%s %s(sec)" % (AnalysisData[pos_cn, 8:10], AnalysisData[pos_cn, 4]))

                    plt.xlim(xmin, xmax)
                    plt.ylim(ymin, ymax)
                    plt.legend()
                    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
                    plt.show()

        plt.style.use("seaborn-whitegrid")
        fig = plt.figure(figsize=(15, 7))

        if LR < 1:
            vp2d = fig.add_subplot(121)
        else:
            vp2d = fig.add_subplot(122)

        vp2d.plot(AnalysisData[1:-1, 0], AnalysisData[1:-1, 1], "o", color="limegreen", markersize=15, label="OK")

        if FalseRT.size != 0:
            vp2d.plot(AnalysisData[FalseRT, 0], AnalysisData[FalseRT, 1], "ro", markersize=15,
                      label="Abnormal RT %s%%" % (np.round(FalseRT.size / chkidx.size * 100, 2)))

        if SlowTOF.size != 0:
            vp2d.plot(AnalysisData[SlowTOF, 0], AnalysisData[SlowTOF, 1], "o", color="gold",
                      label="Slow TOF %s%%" % (np.round(SlowTOF.size / chkidx.size * 100, 2)))

        if Blink.size != 0:
            vp2d.plot(AnalysisData[Blink, 0], AnalysisData[Blink, 1], "b.", label="Blink")

        plt.xticks(list(filter(lambda x: x % 3 == 0, np.arange(-30, 30))))  # 3刻みに目盛り表示
        plt.yticks(list(filter(lambda y: y % 3 == 0, np.arange(-30, 30))))
        plt.legend(bbox_to_anchor=(0.5, -0.1), loc="center", borderaxespad=0, ncol=3)

        if LR < 1:
            plt.xlim(27, -30)
            plt.title("左眼", fontproperties=fp, fontsize=13)
        else:
            plt.xlim(-30, 27)
            plt.title("右眼", fontproperties=fp, fontsize=13)

        cid = fig.canvas.mpl_connect("button_press_event", onclick)

        plt.show()

    # Rt and Tof -------------------------------------------------------------------------------------
    def RtandTofPlot(self):

        mc2 = move_point
        deltamax = xylim

        if "L.csv" in file_name:
            LR = 0
        else:
            LR = 1

        FalseRT = np.where((AnalysisData[1:-1, 6] == 1))[0] + 1
        SlowTOF = np.where((AnalysisData[1:-1, 7] == 1))[0] + 1
        Blink = np.where((AnalysisData[1:-1, 5] > 0.18))[0] + 1

        # d_abs-------------------------------------------------------------------------
        def onclick(event):
            click_x = round(event.xdata, 2)
            click_y = round(event.ydata, 2)
            pos_cn = np.where((AnalysisData[:, 2] == click_x) & (AnalysisData[:, 3] == click_y))[0]

            if pos_cn.size == 0:
                pos_cn = \
                    np.where((AnalysisData[:, 2] <= click_x + 0.01) & (AnalysisData[:, 2] >= click_x - 0.01)
                             & (AnalysisData[:, 3] <= click_y + 0.01) & (AnalysisData[:, 3] >= click_y - 0.01))[0]

            if pos_cn.size != 0:
                # Check False point-----------------------------------
                pos_fnum = 0
                pos_snum = 0
                pos_bnum = 0

                for i in range(pos_cn.size):
                    chk_fp = np.where(FalseRT == pos_cn[i])[0]
                    if chk_fp.size != 0:
                        pos_fnum = FalseRT[chk_fp]

                for j in range(pos_cn.size):
                    chk_sp = np.where(SlowTOF == pos_cn[j])[0]
                    if chk_sp.size != 0:
                        pos_snum = SlowTOF[chk_sp]

                for k in range(pos_cn.size):
                    chk_bp = np.where(Blink == pos_cn[k])[0]
                    if chk_bp.size != 0:
                        pos_bnum = Blink[chk_bp]

                if pos_fnum != 0:
                    pos_num = pos_fnum
                elif pos_snum != 0:
                    pos_num = pos_snum
                elif pos_bnum != 0:
                    pos_num = pos_bnum
                else:
                    pos_num = pos_cn[-1]

                pos_cn = int(pos_num)

                if pos_cn == chkidx.size - 1:
                    cal_xy = pos[chkidx[pos_cn]:-1, ]
                    cal_mc2 = mc2[chkidx[pos_cn]:-1, ]
                else:
                    cal_xy = pos[chkidx[pos_cn]:chkidx[pos_cn + 1], ]
                    cal_mc2 = mc2[chkidx[pos_cn]:chkidx[pos_cn + 1], ]

                # find eye move
                findmove = np.where(cal_mc2 > 0)[0]
                findfix = np.where(cal_mc2 == 0)[0]
                if pos_bnum != 0:
                    findblink = np.where(cal_mc2 == 2)[0]

                if findfix.size > 0:
                    fix_old = findfix.size - 1
                    fix_new = findfix[fix_old]
                    for fn in range(findfix.size - 1):
                        if findfix[fn + 1] != findfix[fn] + 1:
                            fix_old = findfix[fn] + 1
                            fix_new = findfix[fn + 1:-1]
                            break

                plt.style.use("seaborn")
                if LR < 1:
                    gp2d = fig.add_subplot(122)
                else:
                    gp2d = fig.add_subplot(121)

                gp2d.cla()

                # Fix max min ----------------------------------------------
                meanxy = np.mean(cal_xy, axis=0)
                maxmax = np.amax(deltamax)
                xmin = meanxy[0] - maxmax
                xmax = meanxy[0] + maxmax
                ymin = meanxy[1] - maxmax
                ymax = meanxy[1] + maxmax

                """
                # Movie-----------------------------------------------------------
                plt.title("%s \n (%s -> %s)" % (AnalysisData[pos_cn, 0:2], AnalysisData[pos_cn-1, 8:10], AnalysisData[pos_cn, 8:10]), fontdict={"fontsize": 13})
                plotsize=chkidx[pos_cn + 1] - chkidx[pos_cn] + 1
                for num in range(plotsize):
                plt.cla()
                gp2d.plot(cal_xy[0:num,0], cal_xy[0:num,1], color="deepskyblue")
                gp2d.plot(cal_xy[num-1,0], cal_xy[num-1,1], "o", color="deepskyblue")
                plt.xlim(xmin, xmax)
                plt.ylim(ymin, ymax)
                plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
                plt.pause(0.01)
                """

                # plot-----------------------------------------------------------
                if pos_cn != 0:
                    plt.title("No.%s %s \n (%s -> %s)" % (pos_cn,
                    AnalysisData[pos_cn, 0:2], AnalysisData[pos_cn - 1, 8:10], AnalysisData[pos_cn, 8:10]),
                              fontdict={"fontsize": 13})
                    gp2d.plot(cal_xy[:, 0], cal_xy[:, 1], "-", color="deepskyblue")
                    gp2d.plot(cal_xy[findfix[0]:fix_old, 0], cal_xy[findfix[0]:fix_old, 1], "o", color="gold",
                              label="Reaction Time(RT)%s(sec)" % (AnalysisData[pos_cn, 2]))
                    gp2d.plot(cal_xy[findmove, 0], cal_xy[findmove, 1], ".", color="deepskyblue",
                              label="Time of Flight(ToF)%s(sec)" % (AnalysisData[pos_cn, 3]))
                    if pos_bnum != 0:
                        gp2d.plot(cal_xy[findblink, 0], cal_xy[findblink, 1], "b*", label="Blink")

                    gp2d.plot(cal_xy[fix_new, 0], cal_xy[fix_new, 1], "ro",
                              label="fixation%s %s(sec)" % (AnalysisData[pos_cn, 8:10], AnalysisData[pos_cn, 4]))

                    plt.xlim(xmin, xmax)
                    plt.ylim(ymin, ymax)
                    plt.legend()
                    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
                    plt.show()

        plt.style.use("seaborn-whitegrid")
        fig = plt.figure(figsize=(17, 8))

        if LR < 1:
            rf2d = fig.add_subplot(121)
        else:
            rf2d = fig.add_subplot(122)

        rf2d.plot(AnalysisData[1:-1, 2], AnalysisData[1:-1, 3], "o", color="limegreen", label="OK")
        if FalseRT.size != 0:
            rf2d.plot(AnalysisData[FalseRT, 2], AnalysisData[FalseRT, 3], "ro", label="Abnormal RT")
        if SlowTOF.size != 0:
            rf2d.plot(AnalysisData[SlowTOF, 2], AnalysisData[SlowTOF, 3], ".", color="gold", label="Slow ToF")
        if Blink.size != 0:
            rf2d.plot(AnalysisData[Blink, 2], AnalysisData[Blink, 3], "b.", label="Blink")

        plt.legend(bbox_to_anchor=(0.5, -0.1), loc="center", borderaxespad=0, ncol=4)

        timemax = np.amax(AnalysisData[1:-1, 2:4])
        if timemax < 3:
            timemax = 3

        plt.xlim(-0.05, timemax * 1.1)
        plt.ylim(-0.05, timemax * 1.1)

        if LR < 1:
            plt.title("左眼", fontproperties=fp, fontsize=13)
        else:
            plt.title("右眼", fontproperties=fp, fontsize=13)

        cid = fig.canvas.mpl_connect("button_press_event", onclick)
        rf2d.set_ylabel("Time of Flight(sec)")
        rf2d.set_xlabel("Reaction Time(sec)")

        plt.show()

    # TimeData -------------------------------------------------------------------------------------
    def TimedataPlot(self):

        if "L.csv" in file_name:
            LR = 0
        else:
            LR = 1


        plt.style.use("seaborn-whitegrid")
        fig_t = plt.figure(figsize=(15, 7))

        signal_x = fig_t.add_subplot(211)
        signal_y = fig_t.add_subplot(212)

        rt = np.mean(AnalysisData[:, 2])
        samplingfreq = StartPage.sampling_frequency(self)

        signal_x.plot(signal[:, 0] / samplingfreq, pos[:, 0], "b.-", label="Horizontal Signal")
        ax2 = signal_x.twinx()
        ax2.step(AnalysisData[:, 10] + rt, AnalysisData[:, 0], "g.-", where="post", label="Real degree(X)")

        signal_y.plot(signal[:, 0] / samplingfreq, pos[:, 1], "r.-", label="Vertical Signal")
        ax3 = signal_y.twinx()
        ax3.step(AnalysisData[:, 10] + rt, AnalysisData[:, 1], "g.-", where="post", label="Real degree(Y)")

        signal_y.set_xlabel("Time(sec)")
        signal_x.set_ylabel("Signal")
        signal_y.set_ylabel("Signal")

        if LR < 1:
            signal_x.set_title("Left Horizontal Signal")
            signal_y.set_title("Left Vertical Signal")
        else:
            signal_x.set_title("Right Horizontal Signal")
            signal_y.set_title("Right Vertical Signal")

        signal_x.legend()
        signal_y.legend()
        plt.show()

app = ShowViewPointapp()
app.mainloop()
