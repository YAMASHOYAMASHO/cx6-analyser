import SmartLab
import tkinter as tk
from tkinter import ttk
import os
import json
from tkinter import filedialog


class App:
    """
    Appクラス
    このクラスはSmartLabデータと対話するためのグラフィカルユーザーインターフェース(GUI)を提供します。
    データの読み込み、ステータスファイルの作成、SmartLabシステムからのデータ取得などの機能を含みます。
    GUIはtkinterライブラリを使用して構築されています。
    主な機能:
    - 新規測定データの読み込み，解析
    - ステータスファイルの作成
    - データフォルダの取得

    メソッド
    -------
    read_data():
        SmartLabのステータスファイルとデータファイルを選択して読み込むためのGUIを表示します。
        新規のデータ解析用のメゾットです。
    make_status():
        ユーザーが提供したパラメータを使用して新しいSmartLabステータスファイルを作成するためのGUIを表示します。
    get_datas():
        SmartLabデータフォルダを選択して取得するためのGUIを表示します。

    プライベートメソッド
    -----------------
    _read_data_x():
        選択されたステータスファイルに基づいてデータファイルを読み込むロジックを処理します。
    _make_status_x():
        新しいSmartLabステータスファイルを作成して保存するロジックを処理します。
    _get_datas():
        検索とフィルタリングオプションを備えたデータフォルダ選択用のメインGUIを表示します。
    _display_checkboxes(filtered_paths):
        指定されたフォルダパスのリストに基づいてチェックボックスを表示します。
    _filter_data():
        検索入力に基づいて表示されるチェックボックスをフィルタリングします。
    _get_data_x():
        選択されたデータフォルダを取得し、SmartLabデータオブジェクトを初期化します。
    """

    def read_data(self):
        """
        ステータスファイルと測定データファイルを選択して新規データを解析するGUIを表示します。
        """
        self.root = tk.Tk()
        self.root.title("SmartLab Read Data")
        self.root.geometry("400x300")
        self.root.configure(bg="#f0f0f0")

        # タイトルラベルを作成して配置
        title_label = tk.Label(
            self.root, text="Select a Status File", font=("Helvetica", 16), bg="#f0f0f0"
        )
        title_label.pack(pady=10)

        # ステータスファイルのリストを取得
        status_path = os.listdir("datas/status")
        status_path = [x for x in status_path if x.endswith(".json")]

        # ラジオボタンを作成して配置
        self.status_var = tk.StringVar()
        if status_path:
            self.status_var.set(status_path[0])  # 初期値を設定

        for path in status_path:
            rb = tk.Radiobutton(
                self.root, text=path, variable=self.status_var, value=path, bg="#f0f0f0"
            )
            rb.pack(anchor=tk.W, padx=20)

        # データ読み込みボタンを作成して配置
        self.button = tk.Button(
            self.root,
            text="Read Data",
            command=self._read_data_x,
            bg="#4CAF50",
            fg="white",
        )
        self.button.pack(pady=20)

        self.root.mainloop()

    def _read_data_x(self):
        """
        選択されたステータスファイルを基にデータファイルを読み込みます。
        プログレスバーと結果収集システムを使用して一括処理を行います。
        """
        # 選択されたステータスファイルを読み込む
        status_path = self.status_var.get()
        with open("datas/status/" + status_path) as f:
            status = json.load(f)

        # ファイルダイアログを表示してファイルを選択
        paths = filedialog.askopenfilenames()
        
        if not paths:
            return

        # GUIを更新して処理状況を表示
        for widget in self.root.winfo_children():
            if isinstance(widget, (tk.Button, tk.Radiobutton)):
                widget.destroy()

        # 処理状況表示エリア
        status_frame = tk.Frame(self.root, bg="#f0f0f0")
        status_frame.pack(pady=20, fill=tk.BOTH, expand=True)

        status_label = tk.Label(
            status_frame, 
            text="ファイル処理中...", 
            font=("Helvetica", 14), 
            bg="#f0f0f0"
        )
        status_label.pack(pady=(0, 10))

        # 現在のファイル表示
        current_file_var = tk.StringVar()
        current_file_label = tk.Label(
            status_frame, 
            textvariable=current_file_var, 
            bg="#f0f0f0",
            wraplength=350
        )
        current_file_label.pack(pady=(0, 10))

        # プログレスバー
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(
            status_frame, 
            variable=progress_var, 
            maximum=len(paths),
            length=350
        )
        progress_bar.pack(fill=tk.X, padx=20, pady=(0, 10))

        # 統計情報表示
        stats_var = tk.StringVar()
        stats_label = tk.Label(
            status_frame,
            textvariable=stats_var,
            bg="#f0f0f0",
            font=("Helvetica", 10)
        )
        stats_label.pack()

        # 結果収集オブジェクト
        result = SmartLab.ProcessingResult()

        # ファイルを順番に処理
        for i, path in enumerate(paths):
            filename = os.path.basename(path)
            
            # GUI更新
            current_file_var.set(f"処理中: {filename}")
            progress_var.set(i)
            stats_var.set(f"成功: {len(result.successes)}  エラー: {len(result.errors)}")
            self.root.update()

            try:
                # SmartLab_dataオブジェクトの作成と処理
                data = SmartLab.SmartLab_data()
                data.set_status(status, logging=lambda x: None)  # 静音
                data.read_txt(path, logging=lambda x: None, result_collector=result)
                data.save_datas(
                    path="datas/SmartLab", 
                    name=filename,
                    logging=lambda x: None,  # 静音
                    result_collector=result
                )
                
                # 成功時のコンソール出力（重要な情報のみ）
                print(f"✓ {filename}: {data.r:.1f} ± {data.ci_r:.1f} MPa")

            except Exception as e:
                # エラーは既にresult_collectorで収集済み
                print(f"✗ {filename}: {str(e)}")

        # 最終更新
        progress_var.set(len(paths))
        current_file_var.set("処理完了")
        stats_var.set(f"最終結果 - 成功: {len(result.successes)}  エラー: {len(result.errors)}")
        self.root.update()

        # 結果サマリーをコンソールに表示
        result.print_summary()

        # 完了ボタンを表示
        finish_button = tk.Button(
            status_frame,
            text="完了",
            command=self.root.destroy,
            bg="#4CAF50",
            fg="white",
            font=("Helvetica", 12)
        )
        finish_button.pack(pady=20)

    def make_status(self):
        """
        新しいステータスファイルを作成するためのGUIを表示します。
        """
        self.root = tk.Tk()
        self.root.title("SmartLab Make Status")
        self.root.geometry("400x400")
        self.root.configure(bg="#f0f0f0")

        # タイトルラベルを作成して配置
        title_label = tk.Label(
            self.root, text="Create a New Status", font=("Helvetica", 16), bg="#f0f0f0"
        )
        title_label.pack(pady=10)

        # 入力フィールドを作成して配置
        fields = [
            ("name", ""),
            ("poisson", ""),
            ("modulus", ""),
            ("wavelength_SmartLab", ""),
            ("ψ_SmartLab", ""),
            ("φ_SmartLab", ""),
        ]

        self.entries = {}
        for field, default in fields:
            label = tk.Label(self.root, text=field, bg="#f0f0f0")
            label.pack()
            entry = tk.Entry(self.root)
            entry.insert(0, default)
            entry.pack()
            self.entries[field] = entry

        # ステータス作成ボタンを作成して配置
        self.button = tk.Button(
            self.root,
            text="Make Status",
            command=self._make_status_x,
            bg="#4CAF50",
            fg="white",
        )
        self.button.pack(pady=20)
        self.root.mainloop()

    def _make_status_x(self):
        """
        入力されたデータを基にステータスファイルを作成して保存します。
        """
        # 入力されたデータを取得
        name = self.entries["name"].get()
        poisson = float(self.entries["poisson"].get())
        modulus = float(self.entries["modulus"].get())
        wavelength_SmartLab = float(self.entries["wavelength_SmartLab"].get())
        psi_SmartLab = [float(x) for x in self.entries["ψ_SmartLab"].get().split(",")]
        phi_SmartLab = [float(x) for x in self.entries["φ_SmartLab"].get().split(",")]

        # ステータスを作成して保存
        path = "datas/status/"
        SmartLab.make_status(
            path=path,
            name=name,
            poisson=poisson,
            modulus=modulus,
            wavelength_SmartLab=wavelength_SmartLab,
            psi_SmartLab=psi_SmartLab,
            phi_SmartLab=phi_SmartLab,
        )

        # ウィンドウを閉じる
        self.root.destroy()

    def _get_datas(self):
        """
        データフォルダを選択し、検索やフィルタリングを行うためのGUIを表示します。
        """
        self.root = tk.Tk()
        self.root.title("SmartLab Get Data")
        self.root.geometry("600x500")
        self.root.configure(bg="#f0f0f0")

        # タイトルラベルを作成して配置
        title_label = tk.Label(
            self.root, text="Select Data", font=("Helvetica", 16), bg="#f0f0f0"
        )
        title_label.pack(pady=10)

        # 検索フレームを作成して配置
        search_frame = tk.Frame(self.root, bg="#f0f0f0")
        search_frame.pack(fill=tk.X, padx=10, pady=5)

        search_label = tk.Label(search_frame, text="Search:", bg="#f0f0f0")
        search_label.pack(side=tk.LEFT, padx=5)

        self.search_var = tk.StringVar()
        search_entry = tk.Entry(search_frame, textvariable=self.search_var)
        search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        search_button = tk.Button(
            search_frame,
            text="Search",
            command=self._filter_data,
            bg="#4CAF50",
            fg="white",
        )
        search_button.pack(side=tk.LEFT, padx=5)

        # データ表示フレームを作成して配置
        frame = tk.Frame(self.root, bg="#f0f0f0")
        frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(frame, bg="#f0f0f0")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )

        def on_mouse_wheel(event):
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        self.canvas.bind_all("<MouseWheel>", on_mouse_wheel)

        self.checkbox_frame = tk.Frame(self.canvas, bg="#f0f0f0")
        self.canvas.create_window((0, 0), window=self.checkbox_frame, anchor="nw")

        # データフォルダのリストを取得
        self.paths = os.listdir("datas/SmartLab")
        self.paths = [
            x for x in self.paths if os.path.isdir(os.path.join("datas/SmartLab", x))
        ]
        self._display_checkboxes(self.paths)

        # データ取得ボタンを作成して配置
        self.button = tk.Button(
            self.root,
            text="Get Data",
            command=self._get_data_x,
            bg="#4CAF50",
            fg="white",
        )
        self.button.pack(pady=20)

        self.root.mainloop()

    def _display_checkboxes(self, filtered_paths):
        """
        指定されたフォルダパスのリストに基づいてチェックボックスを表示します。

        Parameters
        ----------
        filtered_paths : list
            表示するフォルダパスのリスト
        """
        # 既存のチェックボックスを削除
        for widget in self.checkbox_frame.winfo_children():
            widget.destroy()

        # 新しいチェックボックスを作成して配置
        self.data_vars = {}
        for path in filtered_paths:
            var = tk.BooleanVar()
            cb = tk.Checkbutton(
                self.checkbox_frame, text=path, variable=var, bg="#f0f0f0"
            )
            cb.pack(anchor=tk.W, padx=20)
            self.data_vars[path] = var

    def _filter_data(self):
        """
        検索条件に基づいて表示するチェックボックスをフィルタリングします。
        """
        # 検索条件に基づいてパスをフィルタリング
        search_terms = self.search_var.get().strip().split()
        filtered_paths = [
            path
            for path in self.paths
            if all(term.lower() in path.lower() for term in search_terms)
        ]
        self._display_checkboxes(filtered_paths)

    def _get_data_x(self):
        """
        選択されたデータフォルダを取得し、SmartLabデータオブジェクトを初期化します。
        """
        # 選択されたパスを取得
        selected_paths = [path for path, var in self.data_vars.items() if var.get()]

        # データを取得して保存
        self.data = [
            SmartLab.SmartLab_data(path=os.path.join("datas/SmartLab", path))
            for path in selected_paths
        ]

        # ウィンドウを閉じる
        self.root.quit()
        self.root.destroy()

    def get_datas(self):
        """
        データフォルダを取得し、SmartLabデータオブジェクトを返します。

        Returns
        ----------
        list
            SmartLabデータオブジェクトのリスト
        """
        self._get_datas()
        return self.data


class Maker:
    """
    SmartLabデータを作成するためのGUIアプリケーション。
    """

    def __init__(self):
        """
        Makerクラスの初期化処理を行います。
        """
        self.root = tk.Tk()
        self.root.title("SmartLab Make Data")
        self.root.geometry("400x400")
        self.root.configure(bg="#f0f0f0")

        # タイトルラベルを作成して配置
        title_label = tk.Label(
            self.root, text="Make Data", font=("Helvetica", 16), bg="#f0f0f0"
        )
        title_label.pack(pady=10)

        # ステータスファイルのリストを取得
        status_path = os.listdir("datas/status")
        status_path = [x for x in status_path if x.endswith(".json")]

        # ラジオボタンを作成して配置
        self.status_var = tk.StringVar()
        for path in status_path:
            rb = tk.Radiobutton(
                self.root, text=path, variable=self.status_var, value=path, bg="#f0f0f0"
            )
            rb.pack(anchor=tk.W, padx=20)

        # データ作成ボタンを作成して配置
        self.button = tk.Button(
            self.root,
            text="Make Data",
            command=self._make_data_x,
            bg="#4CAF50",
            fg="white",
        )
        self.button.pack(pady=20)

    def run(self):
        """
        GUIアプリケーションを開始します。
        """
        self.root.mainloop()

    def _make_data_x(self):
        """
        選択されたステータスファイルを基にデータを作成して保存します。
        """
        # 選択されたステータスファイルを読み込む
        status_path = self.status_var.get()
        with open("datas/status/" + status_path) as f:
            status = json.load(f)

        # ファイルダイアログを表示してファイルを選択
        paths = filedialog.askopenfilenames()

        # ラベルを作成して配置
        label = tk.Label(self.root, text="", bg="#f0f0f0")
        label.pack()
        label = ttk.Label(self.root, text="Making Data...", background="#f0f0f0")
        label.pack()
        textvar = tk.StringVar()
        label = ttk.Label(self.root, textvariable=textvar, background="#f0f0f0")
        label.pack()

        # 選択されたファイルを読み込んで処理
        for path in paths:
            try:
                textvar.set(path.split("/")[-1])
                data = SmartLab.SmartLab_data()
                data.set_status(status)
                data.read_txt(path)
                data.save_datas(path="datas/SmartLab", name=os.path.basename(path))
            except Exception as e:
                print(path + " is not read")
                print(e)

        # ウィンドウを閉じる
        self.root.destroy()
