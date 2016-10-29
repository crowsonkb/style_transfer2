/* The main JavaScript file for the Style Transfer application. */

$(document).ready(function() {
    // Refresh the output image every five seconds.
    var update_every = 5000;
    window.setInterval(function() {
        $("#output-image").attr("src", "/output.png");
    }, update_every);
});
