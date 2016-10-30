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

function upload(slot) {
    var reader = new FileReader();
    reader.onload = function(e) {
        var size = $("#resize-to").val()
        var msg = {size: size, slot: slot, data: e.target.result};
        $.post("/upload", msg);
    };
    reader.readAsDataURL($("#file-selector")[0].files[0])
}

$(document).ready(function() {
    // Refresh the output image every second.
    var update_every = 1000;
    window.setInterval(function() {
        $("#output-image").attr("src", "/output.png");
    }, update_every);

    function ws_connect() {
        ws = new WebSocket("ws://" + window.location.host + "/websocket");

        ws.onmessage = function(e) {
            var msg = JSON.parse(e.data);

            switch (msg.type) {
            case "iterateInfo":
                $("#iterate").text(msg.i);
                $("#loss").text(msg.loss.toPrecision(4));
                $("#step-size").text(msg.stepSize.toPrecision(3));
                $("#its-per-s").text(msg.itsPerS.toPrecision(3));
                break;
            case "newParams":
                $("#params").val(msg.params);
                break;
            }
        };

        ws.onclose = function(e) { ws_connect(); };
    }
    ws_connect();
});
