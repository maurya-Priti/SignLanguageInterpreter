function startCamera() {
    document.getElementById('camera-feed').style.display = 'block';
    document.getElementById('video').src = "/video_feed" ;
}

function stopCamera() {
    document.getElementById('camera-feed').style.display = 'none';
    document.getElementById('video').src = "";
}
    