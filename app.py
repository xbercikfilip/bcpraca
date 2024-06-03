from flask import Flask, send_from_directory, jsonify, request
from projekt.show.show import getRecommendations 

app = Flask(__name__)

@app.route("/")
def index():
    return send_from_directory("projekt", "index.html")

@app.route("/<path:filename>")
def serve_static(filename):
    return send_from_directory("projekt", filename)

@app.route('/getImages', methods=['POST'])
def update_images():
    liked_images = request.json['likedImages']
    mode = request.json['mode']

    print(liked_images)
    print(mode)
    images, original  = getRecommendations(liked_images, mode)
    if images:
        data = {"message": "ok", "images" : images, "original" : original}
    else:
        data = {"message": "failed"}

    return jsonify(data)


if __name__ == '__main__':
    app.run(debug=True)


# flask --app server run