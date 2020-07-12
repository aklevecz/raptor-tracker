let mobilenet;
let model;
const videoElement = document.getElementById("wc");
videoElement.width = window.innerWidth * 1.15;
videoElement.height = window.innerHeight;
const webcam = new Webcam(videoElement);
const dataset = new RPSDataset();
let raptorSamples = 0;
let porperSamples = 0;
let isPredicting = false;

async function loadMobilenet() {
  const mobilenet = await tf.loadLayersModel(
    "https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json"
  );
  const layer = mobilenet.getLayer("conv_pw_13_relu");
  console.log(layer);
  return tf.model({ inputs: mobilenet.inputs, outputs: layer.output });
}

async function loadModel() {
  return await tf.loadLayersModel("/modeljs/model.json");
}

async function train() {
  dataset.ys = null;
  const numLabels = 2;
  dataset.encodeLabels(numLabels);
  model = tf.sequential({
    layers: [
      tf.layers.flatten({ inputShape: mobilenet.outputs[0].shape.slice(1) }),
      tf.layers.dense({ units: 100, activation: "relu" }),
      tf.layers.dense({ units: numLabels, activation: "sigmoid" }),
    ],
  });
  const optimizer = tf.train.adam(0.0001);
  model.compile({ optimizer, loss: "categoricalCrossentropy" });
  let loss = 0;
  model.fit(dataset.xs, dataset.ys, {
    epochs: 10,
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        loss = logs.loss.toFixed(5);
        console.log("LOSS:" + loss);
      },
    },
  });
}
window.addEventListener("keydown", (e) => {
  console.log(e.key);
  if (e.key == "r") {
    handleButton({ id: "0" });
  } else {
    handleButton({ id: "1" });
  }
});
function handleButton(elem) {
  console.log(elem);
  switch (elem.id) {
    case "0":
      raptorSamples++;
      document.getElementById("raptorSamples").innerText =
        "raptor samples:" + raptorSamples;
      break;
    case "1":
      porperSamples++;
      document.getElementById("porperSamples").innerText =
        "porper samples:" + porperSamples;
      break;
  }
  label = parseInt(elem.id);
  const img = webcam.capture();
  dataset.addExample(mobilenet.predict(img), label);
}

async function predict() {
  while (isPredicting) {
    const predictedClass = tf.tidy(() => {
      const img = webcam.capture();
      // const activation = mobilenet.predict(img);
      const predictions = model.predict(img);
      // window.predictions = predictions;
      // predictions.print();
      return predictions.as1D();
    });
    const classId = (await predictedClass.data())[0];
    console.log(classId);
    // console.log(await predictedClass.data());
    let predictionText = "";
    predictionText = classId === 0 ? "I see nothing" : classId.toFixed(5);
    if (classId > 0.5) {
      webcam.setKernal("edge");
    } else {
      webcam.setKernal("normal");
    }
    // switch (classId) {
    //   case 0:
    //     predictionText = "I see raptor";
    //     break;
    //   case 1:
    //     predictionText = "I see porper";
    //     break;
    // }

    document.getElementById("prediction").innerText = predictionText;
    predictedClass.dispose();
    await tf.nextFrame();
  }
}

async function doTraining() {
  train();
}

function startPredicting() {
  isPredicting = true;
  predict();
}

function stopPredicting() {
  isPredicting = false;
  predict();
}

function saveModel() {
  model.save("downloads://newd_model");
}

async function init() {
  await webcam.setup();
  // mobilenet = await loadMobilenet();
  //  tf.tidy(() => mobilenet.predict(webcam.capture()));

  model = await loadModel();
  startPredicting();
  tf.tidy(() => model.predict(webcam.capture()));
}

init();
