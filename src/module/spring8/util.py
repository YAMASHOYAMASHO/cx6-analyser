"""
2025.11.08時点におけるファイル抽出プログラム
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import pathlib
import os

DEFAULT_DIR = "//nishinolab-data/Public/SPring-8"


def select_deep_files():
    """
    1. filedialog.askdirectoryで親ディレクトリを選択。
    2. フォルダ一覧をチェックボタン（ラジオボタン風）で表示。
    3. 決定または再選択ボタンで処理を制御。
    4. 決定後、選択されたフォルダ内のファイルリスト（ダミー）を返却。

    Returns:
        list: 選択されたフォルダ内のファイルパスのリスト (現在はダミー)
    """

    # --- 内部状態変数 ---
    parent_dir = ""  # 親ディレクトリパス
    folder_list = []  # 直下のフォルダ一覧
    checkboxes = {}  # {フォルダ名: tk.IntVar} の辞書

    # ウィジェットの状態を制御するための変数
    result_action = None  # 'confirm' (決定) または 'restart' (再選択) を格納

    root = tk.Tk()
    root.title("フォルダ選択ウィジェット")
    # root.geometry("400x350") # サイズ指定を省略し、中身に合わせる

    # 処理完了までウィンドウを隠さないようにするため、メインループを自前で制御
    root.withdraw()

    # --- ステップ1: 親ディレクトリ選択 (自動召喚) ---
    parent_dir = filedialog.askdirectory(title="1. 親ディレクトリを選択してください")
    if not parent_dir:
        root.destroy()
        return []  # キャンセルされた場合は空リストを返す

    # ステップ2: フォルダ一覧を取得
    try:
        # os.listdirでフォルダの一覧を取得
        folder_list = [
            d
            for d in os.listdir(parent_dir)
            if os.path.isdir(os.path.join(parent_dir, d))
        ]
    except Exception as e:
        print(f"ディレクトリ一覧の取得中にエラーが発生しました: {e}")
        root.destroy()
        return []

    # フォルダがない場合は終了
    if not folder_list:
        messagebox.showinfo(
            "情報", "選択したディレクトリにサブフォルダがありませんでした。"
        )
        root.destroy()
        return []

    # メインウィンドウを表示
    root.deiconify()

    # --- ステップ3: ラジオボタン風ウィジェットの作成 ---

    # スクロールバーのためのキャンバスとフレーム
    canvas = tk.Canvas(root, height=200, width=380)
    scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    # マウスホイールでスクロールできるようにする
    def on_mousewheel(event):
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    canvas.bind_all("<MouseWheel>", on_mousewheel)

    tk.Label(root, text=f"2. 選択可能なフォルダ一覧: {parent_dir}").pack(pady=5)

    # フォルダごとのチェックボタンを作成
    for folder_name in folder_list:
        # フォルダごとの状態を保持するIntVar (複数選択のためCheckbuttonを採用)
        var = tk.IntVar(value=0)
        cb = tk.Checkbutton(scrollable_frame, text=folder_name, variable=var)
        cb.pack(anchor="w", padx=10)
        checkboxes[folder_name] = var  # フォルダ名と変数を紐付けて保存

    # 配置
    scrollbar.pack(side="right", fill="y")
    canvas.pack(side="top", fill="both", expand=True, padx=5, pady=5)

    # --- 決定と再選択のボタン ---

    def confirm_action():
        """決定ボタンのイベントハンドラ"""
        nonlocal result_action
        selected_count = sum(var.get() for var in checkboxes.values())
        if selected_count == 0:
            messagebox.showwarning("警告", "フォルダを1つ以上選択してください。")
            return
        result_action = "confirm"
        root.quit()  # メインループを終了

    def restart_action():
        """再選択ボタンのイベントハンドラ"""
        nonlocal result_action
        result_action = "restart"
        root.quit()  # メインループを終了

    button_frame = tk.Frame(root)
    button_frame.pack(pady=10)

    tk.Button(
        button_frame, text="決定 (選択したフォルダを処理)", command=confirm_action
    ).pack(side="left", padx=10)
    tk.Button(
        button_frame, text="再選択 (ディレクトリからやり直す)", command=restart_action
    ).pack(side="left", padx=10)

    # ウィンドウが表示されている間、メインループを実行
    root.mainloop()

    # root.quit()で終了した後、root.destroy()でウィンドウを破棄
    root.destroy()

    # --- 再選択処理 ---
    if result_action == "restart":
        # 再帰呼び出しでやり直す
        return select_deep_files()

    # --- 決定後の処理 ---
    if result_action == "confirm":
        # 選択されたフォルダ名を取得
        selected_folders = [name for name, var in checkboxes.items() if var.get() == 1]

        # 4. 選択したフォルダの奥の方のファイルをリストとして取得，返り値に設定
        # --- ユーザー設計領域 (コメントアウト指定部分) ---

        final_paths = []  # 最終的なファイルパスリスト

        for folder_name in selected_folders:
            target_path = pathlib.Path(parent_dir) / folder_name
            target_path = target_path / "Pilatus"
            paths = list(
                target_path.rglob("*.tif")
            )  # 例: Pilatusフォルダ内の.tifファイルを再帰的に取得

            final_paths.extend([str(p) for p in paths])

        return final_paths

    return []  # 何もアクションがなかった場合

def select_multiple_subdirectories(parent_dir=None):
    """
    ListboxとOSダイアログを組み合わせ、ユーザーに複数のサブフォルダを選択させる関数。

    ステップ1: filedialogで親となるディレクトリを選択させる。
    ステップ2: 選択された親ディレクトリ内のサブフォルダをListboxに表示する。
    ステップ3: ListboxでユーザーがCtrl/Shiftクリックで複数選択し、OKボタンで確定する。
    ステップ4: 選択されたフォルダの完全なパスリストを返す。

    Returns:
        list: 選択されたサブフォルダの完全なパスのリスト。キャンセルされた場合は空のリスト。
    """
    # 1. 親フォルダの選択 (filedialogを使用)
    if parent_dir is None:
        parent_dir = filedialog.askdirectory(title="ステップ1: 親フォルダを選択してください")
    if not parent_dir:
        print("親フォルダの選択がキャンセルされました。")
        return []

    # 2. サブフォルダのリストを作成
    # 親フォルダ直下にあるフォルダ（サブディレクトリ）のみを抽出
    subdirectories = sorted([
        name for name in os.listdir(parent_dir)
        if os.path.isdir(os.path.join(parent_dir, name))
    ])
    
    if not subdirectories:
        print(f"'{parent_dir}' の中にサブフォルダが見つかりませんでした。")
        return []
    
    # 3. Listboxウィンドウの作成と表示
    
    # Listboxの選択結果を格納するリスト
    selected_subdirs = []

    def on_ok():
        """OKボタンが押されたときの処理"""
        # Listboxで選択されているインデックスを取得
        selected_indices = listbox.curselection()
        
        # インデックスからフォルダ名を取得し、完全なパスに変換して格納
        for i in selected_indices:
            folder_name = listbox.get(i)
            full_path = os.path.join(parent_dir, folder_name)
            selected_subdirs.append(full_path)
        
        # ウィンドウを閉じる
        top.destroy()

    def on_cancel():
        """キャンセルボタンが押されたときの処理"""
        # 選択リストを空にしたままウィンドウを閉じる (selected_subdirsは空のまま)
        top.destroy()
        
    # ウィンドウのセットアップ
    top = tk.Toplevel(bg="#f0f0f0") # Toplevelで新しいウィンドウを作成
    top.title(f"ステップ2: '{os.path.basename(parent_dir)}'内のフォルダを選択")
    
    # 説明ラベル
    label = tk.Label(
        top, 
        text="CtrlキーまたはShiftキーを押しながらクリックして複数選択してください。",
        font=('Helvetica', 10),
        bg="#f0f0f0",
        padx=10, 
        pady=5
    )
    label.pack()

    # Listboxとスクロールバーのコンテナ (Frame)
    frame = tk.Frame(top)
    frame.pack(padx=10, pady=5, fill='both', expand=True)

    # スクロールバー
    scrollbar = tk.Scrollbar(frame, orient=tk.VERTICAL)
    
    # Listboxの作成
    # selectmode=tk.EXTENDED を設定することで、Shift/Ctrlクリックでの複数選択が可能になる
    listbox = tk.Listbox(
        frame,
        selectmode=tk.EXTENDED,
        yscrollcommand=scrollbar.set,
        height=15,
        width=50,
        font=('Helvetica', 10)
    )
    
    scrollbar.config(command=listbox.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    listbox.pack(side=tk.LEFT, fill='both', expand=True)
    
    # Listboxにサブフォルダ名を追加
    for folder_name in subdirectories:
        listbox.insert(tk.END, folder_name)

    # ボタンのコンテナ
    button_frame = tk.Frame(top, bg="#f0f0f0")
    button_frame.pack(pady=10)

    # OKボタン
    ok_button = tk.Button(button_frame, text="OK (選択確定)", command=on_ok, width=15, bg="#4CAF50", fg="white", font=('Helvetica', 10, 'bold'))
    ok_button.pack(side=tk.LEFT, padx=10)

    # キャンセルボタン
    cancel_button = tk.Button(button_frame, text="キャンセル", command=on_cancel, width=15, bg="#f44336", fg="white", font=('Helvetica', 10, 'bold'))
    cancel_button.pack(side=tk.LEFT, padx=10)

    # ウィンドウが閉じるまで待機 (Listboxの選択が完了するまで処理をブロック)
    top.grab_set() # 他のウィンドウ操作を禁止
    top.wait_window() # ウィンドウが閉じられるまで待機
    
    # 4. 結果を返す
    if selected_subdirs:
        print("\n✅ 選択が完了しました。")
    else:
        print("\n❌ 選択はキャンセルまたはスキップされました。")
        
    return selected_subdirs

# --- 実行部分 ---
if __name__ == "__main__":
    
    # 関数を実行し、結果（選択されたフォルダのリスト）を受け取る
    result_folders = select_multiple_subdirectories()
    
    if result_folders:
        print("\n--- 最終的に返されたフォルダリスト ---")
        print(len(result_folders), "個のフォルダが選択されました。")