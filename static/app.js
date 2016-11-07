/* The main JavaScript file for the Style Transfer application. */

var ws;

function applyParams() {
    var params = $("#params").val();
    var msg = {type: "applyParams", params: params};
    ws.send(JSON.stringify(msg));
}

function reset() {
    ws.send(JSON.stringify({type: "reset"}));
}

function restartWorker() {
    ws.send(JSON.stringify({type: "restartWorker"}));
}

function showTrace() {
    $("#trace").css("display", "");
}

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

function refreshImage() {
    $("#output-image").attr("src", "/output?" + new Date().getTime());
}

function setWithDataURL(url, elem) {
    // Set the inside of the dropzone to a thumbnail
    var img = $("<img>");
    var h, w;
    img[0].onload = function(e) {
        h = img[0].naturalHeight;
        w = img[0].naturalWidth;
        img.attr("class", "replace");
        var scale = parseInt($(elem).css("width")) / Math.max(h, w);
        img.attr("height", h * scale);
        img.attr("width", w * scale);
        $("#" + elem.id + " .replace").replaceWith(img);
    };
    img.attr("src", url);
}

function uploadFile(files, elem, slot) {
    var reader = new FileReader();
    var data = null;
    reader.onload = function(e) {
        var data = e.target.result;
        setWithDataURL(data, elem);
        var size = $("#resize-to").val();
        var msg = {size: size, slot: slot, data: data};
        $.post("/upload", msg);
    };
    reader.readAsDataURL(files[0]);
}

$(document).ready(function() {
    function stopEvent(e) {
        e.stopPropagation(); e.preventDefault();
    }

    function makeDropZone(elem, slot) {
        elem.ondragenter = stopEvent;
        elem.ondragover = stopEvent;
        elem.ondrop = function(e) {
            stopEvent(e);
            $(elem).css("background-color", "rgb(110, 55, 55)");
            setTimeout(function() {
                $(elem).css("background-color", "");
            }, 250);
            uploadFile(e.dataTransfer.files, "#"+elem.id, slot);
        };
    }

    var body = $("body")[0];
    body.ondragenter = stopEvent;
    body.ondragover = stopEvent;
    body.ondrop = stopEvent;
    makeDropZone($("#content-drop")[0], "content");
    makeDropZone($("#style-drop")[0], "style");
    makeDropZone($("#output-image")[0], "input");
    var contentInput = $("#content-input")[0];
    contentInput.onchange = function() {
        uploadFile(this.files, 'content-drop', 'content');
    };
    var styleInput = $("#style-input")[0];
    styleInput.onchange = function() {
        uploadFile(this.files, 'style-drop', 'style');
    };

    // Wait 100ms after loading to refresh output image
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
                    $("#iterate-stats").css("display", "");
                    $("#iterate").text(msg.i);
                    $("#step-size").text(msg.stepSize.toPrecision(3));
                    $("#its-per-s").text(msg.itsPerS.toPrecision(3));

                    var trace_str = "";
                    for (key in msg.trace) {
                        trace_str += key + ": " + msg.trace[key].toPrecision(4);
                        trace_str += "<br>";
                    }
                    $("#trace-placeholder").html(trace_str);
                    break;
                case "newParams":
                    $("#params").val(msg.params);
                    $("#params-error").text(msg.errorString);
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
                case "thumbnails":
                    if (msg.content) {
                        setWithDataURL(msg.content, $("#content-drop")[0]);
                    }
                    if (msg.style) {
                        setWithDataURL(msg.style, $("#style-drop")[0]);
                    }
                    break;
                case "workerReady":
                    $("#pre-start-message").css("display", "none");
                    $("button").attr("disabled", null);
                    break;
            }
        };
        ws.onclose = ws_connect;
    }
    ws_connect();
});
