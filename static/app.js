/* The main JavaScript file for the Style Transfer application. */

var ws;

function applyParams() {
    var params = $("#params").val();
    var msg = {type: "applyParams", params: params};
    ws.send(JSON.stringify(msg));
}

function reset() { ws.send(JSON.stringify({type: "reset"})); }

function restartWorker() { ws.send(JSON.stringify({type: "restartWorker"})); }

var isStart = true;
function start() {
    if (isStart) {
        $("#start").text("Pause");
        isStart = false;
        ws.send(JSON.stringify({type: "start"}));
    } else {
        $("#start").text("Start");
        isStart = true;
        ws.send(JSON.stringify({type: "pause"}));
    }
}

function upload(slot) {
    var reader = new FileReader();
    reader.onload = function(e) {
        var size = $("#resize-to").val()
        var msg = {size: size, slot: slot, data: e.target.result};
        $.post("/upload", msg);
    };
    reader.readAsDataURL($("#file-selector")[0].files[0])
}

function refreshImage() { $("#output-image").attr("src", "/output"); }

$(document).ready(function() {
    function stopEvent(e) { e.stopPropagation(); e.preventDefault(); }

    function makeDropZone(elem, slot) {
        elem.ondragenter = stopEvent;
        elem.ondragover = stopEvent;
        elem.ondrop = function(e) {
            stopEvent(e);
            $(elem).css("background-color", "rgb(110, 55, 55)");
            setTimeout(function() {
                $(elem).css("background-color", "");
            }, 250);
            var file = e.dataTransfer.files[0];
            var reader = new FileReader();
            var data = null;
            reader.onload = function(e) {
                var data = e.target.result;
                var size = $("#resize-to").val()
                var msg = {size: size, slot: slot, data: data};
                $.post("/upload", msg);
            };
            reader.readAsDataURL(file);
        };
    }

    var body = $("body")[0]
    body.ondragenter = stopEvent;
    body.ondragover = stopEvent;
    body.ondrop = stopEvent;
    makeDropZone($("#content-drop")[0], 'content');
    makeDropZone($("#style-drop")[0], 'style');
    makeDropZone($("#output-image")[0], 'input');

    // Wait one second after loading to refresh output image
    var update_every = 100;
    $("#output-image").on("load", function() {
        setTimeout(refreshImage, update_every);
    });
    refreshImage();

    function ws_connect() {
        ws = new WebSocket("ws://" + window.location.host + "/websocket");

        ws.onopen = refreshImage;

        ws.onmessage = function(e) {
            var msg = JSON.parse(e.data);

            switch (msg.type) {
            case "iterateInfo":
                $("#iterate-stats").css("display", "block");
                $("#iterate").text(msg.i);
                $("#loss").text(msg.loss.toPrecision(6));
                $("#step-size").text(msg.stepSize.toPrecision(3));
                $("#its-per-s").text(msg.itsPerS.toPrecision(3));
                break;
            case "newParams":
                $("#params").val(msg.params);
                break;
            case "newSize":
                $("#resize-to").val(Math.max(msg.width, msg.height));
                $("#output-image").attr("width", msg.width);
                $("#output-image").attr("height", msg.height);
                break;
            case "state":  // misc things on the page that aren't editable
                if (msg.running) {
                    $("#start").text("Pause");
                    isStart = false;
                } else {
                    $("#start").text("Start");
                    isStart = true;
                }
                break;
            }
        };

        ws.onclose = ws_connect;
    }
    ws_connect();
});
