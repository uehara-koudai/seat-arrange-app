# app.py

from flask import Flask, request, jsonify, render_template
import numpy as np
import random
import time  
from seat_arrange_EMC import seat_arrange_EMC # seat_arrange.py をインポート
from seat_arrange_ExhaustiveSearch import seat_arrange_ExhaustiveSearch # seat_arrange_ExhaustiveSearch.py をインポート
from seat_arrange_genetic import seat_arrange_genetic # seat_arrange_genetic.py をインポート

np.random.seed(42)
random.seed(42)
np.show_config()


def convert_numpy_types(obj):
    """
    NumPy の型 (int64, float64, ndarray) を標準の Python 型 (int, float, list) に変換する
    """
    if isinstance(obj, np.int64) or isinstance(obj, np.int32):  # NumPy の int を Python の int に変換
        return int(obj)
    elif isinstance(obj, np.float64) or isinstance(obj, np.float32):  # NumPy の float を Python の float に変換
        return float(obj)
    elif isinstance(obj, np.ndarray):  # NumPy の配列をリストに変換
        return obj.tolist()
    elif isinstance(obj, list):  # リストの中身を再帰的に変換
        return [convert_numpy_types(x) for x in obj]
    elif isinstance(obj, dict):  # 辞書の中身を再帰的に変換
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    else:
        return obj  # 変換不要な場合はそのまま返す


app = Flask(__name__) #Flaskアプリケーションのインスタンスが作成される．
# laskに現在のファイルがアプリケーションのメインファイルであることを伝え、
# 

@app.route("/")
def index():
    # ここで index.html というファイルを返す
    return render_template("index.html")



@app.route("/arrange", methods=["POST"]) #Flask アプリケーションにおいてエンドポイント（URLパス）を設定し、どのようなHTTPメソッドでアクセスされるかを定義するデコレータ
def arrange():
    """
    ここに JSON データが届く想定:
    {
      "N": 3,
      "seatPreferenceList": [[0,1,2],[0,1,2],[2,0,1]],
      "seatCoordinate": [[0,0],[1,0],[1,1]],
      "M": 1
    }
    みたいな感じで。
    """
    data = request.json # リクエストの JSON データを取得
    N = data["N"] # 人数を辞書から取得
    seatPreferenceList = np.array(data["seatPreferenceList"])
    seatCoordinate = np.array(data["seatCoordinate"])
    M = data["M"]

    # 🔥 時間計測開始
    start_time = time.time()


    #席替えの計算を呼び出す
    if N <= 8:
        result = seat_arrange_ExhaustiveSearch(N, seatPreferenceList, seatCoordinate, M)
        print("全状態探索で最適化")
    else:
        result = seat_arrange_EMC(
            N, seatPreferenceList, seatCoordinate, M,
            L=N,      # レプリカ数 (Nと同じに設定)
            steps=3000,  # MCMCステップ数
            init_method='use_preference'  # 初期化方法 ('use_preference' も試せる)
        )
        print("レプリカ交換モンテカルロ法で最適化")
    # result = seat_arrange_genetic(N, seatPreferenceList, seatCoordinate, M) #遺伝的アルゴリズムを用いた座席最適配置
    # 🔥 実行時間の計測
    elapsed_time = time.time() - start_time
    print(f"🚀 API 処理時間: {elapsed_time:.2f} 秒")

    # 結果を JSON で返す
    return jsonify(convert_numpy_types(result)) # 結果を JSON 形式で返す


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True) #Rende用









