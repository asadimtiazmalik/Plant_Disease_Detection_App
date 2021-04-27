$("#image-selector").change(function () {
    let reader = new FileReader();
    reader.onload = function () {
        let dataURL = reader.result;
        $("#selected-image").attr("src", dataURL);
        $("#prediction-list").empty();
    }
    let file = $("#image-selector").prop("files")[0];
    reader.readAsDataURL(file);
}); 

let model;
(async function () {
    console.log("loading start");
    model = await tf.loadLayersModel("http://localhost:81/tfjs-models/MobileNetV2/model.json");
    console.log("loading complete");
    $(".progress-bar").hide();
})();


$("#predict-button").click(async function() {
    let image = $("#selected-image").get(0);
    let offset = tf.scalar(255.0);
    let tensor = tf.browser.fromPixels(image)
    .resizeNearestNeighbor([224, 224])
    //.sub(offset)
    .toFloat()
    .div(offset)
    .expandDims();


    // More pre-processing to be added here later
   // let meanImageNetRGB = tf.tensor1d([123.68, 116.779, 103.939]);

   // let processedTensor = tensor.sub(meanImageNetRGB)
   //.reverse(2);


    let predictions = await model.predict(tensor).data();

    let top5 = Array.from(predictions)
        .map(function (p, i) {
            return {
                probability: p,
                className: plant_disease_classes[i]
            };
        }).sort(function (a, b) {
            return b.probability - a.probability;
        }).slice(0,6);


$("#prediction-list").empty();
top5.forEach(function (p) {
    $("#prediction-list").append(`<li>${p.className}: ${p.probability.toFixed(6)}</li>`);
});
})
