/* The main JavaScript file for the Style Transfer application. */

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
    // Refresh the output image every five seconds.
    var update_every = 5000;
    window.setInterval(function() {
        $("#output-image").attr("src", "/output.png");
    }, update_every);
});
