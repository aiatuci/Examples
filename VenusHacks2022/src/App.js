import "./App.css";
import Webcam from "react-webcam";
import { useState, useEffect, useRef, useCallback } from "react";
import * as tf from "@tensorflow/tfjs";
import * as poseDetection from "@tensorflow-models/pose-detection";
import "@tensorflow/tfjs-backend-webgl";

const cameraSize = 500;
const threshold = 0.5;

const WebcamCapture = (props) => {
  const webcamRef = useRef(null);
  const capture = useCallback(() => {
    const imageSrc = webcamRef.current.getScreenshot();
    props.setimage(imageSrc);
  }, [webcamRef]);
  return (
    <>
      <Webcam
        audio={false}
        ref={webcamRef}
        height={cameraSize}
        width={cameraSize}
        screenshotFormat="image/jpeg"
      />
      <p className="btn" onClick={capture}>
        Capture photo
      </p>
    </>
  );
};

const App = () => {
  const [cameraOpen, setcameraOpen] = useState(false);
  const [detector, setdetector] = useState();
  const [imagesrc, setimage] = useState("");
  const [prediction, setprediction] = useState(null);

  const loadModel = async () => {
    const model = poseDetection.SupportedModels.MoveNet;
    const posedetector = await poseDetection.createDetector(model);
    setdetector(posedetector);
  };

  const predict = async () => {
    const predictimg = new Image(cameraSize, cameraSize);
    predictimg.src = imagesrc;
    const poses = await detector.estimatePoses(predictimg);
    setprediction(poses);
  };

  useEffect(() => {
    tf.ready().then(() => {
      loadModel();
    });
  }, []);

  const setCameraState = () => {
    console.log("change camera state");
    setcameraOpen(!cameraOpen);
  };
  let prediction_parts = null;
  if (prediction !== null) {
    prediction_parts = prediction[0].keypoints.map((pred, index) => {
      if (pred.score >= threshold) {
        return (
          <li className="resultitem" key={index}>
            {pred.name}
          </li>
        );
      }
    });
  }
  return (
    <div className="App">
      <div className="leftblock">
        <p className="btn" onClick={setCameraState}>
          {cameraOpen ? "Close Browser Camera" : "Open Browser Camera"}
        </p>
        {cameraOpen ? <WebcamCapture setimage={setimage} /> : null}
      </div>
      <div className="rightblock">
        {imagesrc === "" ? null : <img className="img" src={imagesrc} />}
        {imagesrc === "" ? null : (
          <p className="btn" onClick={predict}>
            Predict
          </p>
        )}
        <ul className="resultlist">{prediction_parts}</ul>
      </div>
    </div>
  );
};

export default App;
