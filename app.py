# app.py

from flask import Flask, request, jsonify, render_template
import numpy as np
from seat_arrange import seat_arrange # seat_arrange.py をインポート


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

    #席替えのロジックを呼び出す
    result = seat_arrange(N, seatPreferenceList, seatCoordinate, M)

    # 結果を JSON で返す
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)









