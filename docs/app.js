import {
    FilesetResolver,
    HandLandmarker,
    PoseLandmarker
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const predictionElement =
    document.getElementById("prediction");

const confidenceElement =
    document.getElementById("confidence");

/*
====================================
VARIABLES
====================================
*/

let handLandmarker;
let poseLandmarker;
let recording = false;

let recordedFrames = [];

let predictionBuffer = [];

const FRAMES_TO_RECORD = 30;

const FEATURES_PER_HAND = 63;
const FEATURES_POSE_ARM = 18;

const TOTAL_FEATURES = 144;

/*
====================================
BOTÓN GRABAR
====================================
*/

/*
====================================
CONEXIONES MANO
====================================
*/

const connections = [

    // Pulgar
    [0,1],
    [1,2],
    [2,3],
    [3,4],

    // Índice
    [0,5],
    [5,6],
    [6,7],
    [7,8],

    // Medio
    [0,9],
    [9,10],
    [10,11],
    [11,12],

    // Anular
    [0,13],
    [13,14],
    [14,15],
    [15,16],

    // Meñique
    [0,17],
    [17,18],
    [18,19],
    [19,20],

    // Palma
    [5,9],
    [9,13],
    [13,17]
];

/*
====================================
CÁMARA
====================================
*/

async function setupCamera() {

    const stream =
        await navigator.mediaDevices.getUserMedia({
            video: true
        });

    video.srcObject = stream;

    return new Promise((resolve) => {

        video.onloadedmetadata = () => {

            video.play();

            resolve(video);
        };
    });
}

/*
====================================
MEDIAPIPE
====================================
*/

async function createHandLandmarker() {

    const vision =
        await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
        );

    handLandmarker =
        await HandLandmarker.createFromOptions(
            vision,
            {
                baseOptions: {
                    modelAssetPath:
                        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
                },

                runningMode: "VIDEO",

                numHands: 2
            }
        );
}

async function createPoseLandmarker() {

    const vision =
        await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
        );

    poseLandmarker =
        await PoseLandmarker.createFromOptions(
            vision,
            {
                baseOptions: {
                    modelAssetPath:
                        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
                },
                runningMode: "VIDEO",
                numPoses: 1
            }
        );
}

/*
====================================
DIBUJAR
====================================
*/

function drawPoint(x, y) {

    ctx.beginPath();

    ctx.arc(
        x,
        y,
        5,
        0,
        2 * Math.PI
    );

    ctx.fillStyle = "#ff0000";

    ctx.fill();
}

function drawLine(x1, y1, x2, y2) {

    ctx.beginPath();

    ctx.moveTo(x1, y1);

    ctx.lineTo(x2, y2);

    ctx.strokeStyle = "#00ff99";

    ctx.lineWidth = 3;

    ctx.stroke();
}

function drawHand(landmarks) {

    // Dibujar líneas

    for (const connection of connections) {

        const start =
            landmarks[connection[0]];

        const end =
            landmarks[connection[1]];

        drawLine(
            start.x * canvas.width,
            start.y * canvas.height,
            end.x * canvas.width,
            end.y * canvas.height
        );
    }

    // Dibujar puntos

    for (const point of landmarks) {

        drawPoint(
            point.x * canvas.width,
            point.y * canvas.height
        );
    }
}

function drawPoseArm(poseLandmarks) {

    if (!poseLandmarks) return;

    const points = poseLandmarks[0];

    if (!points) return;

    const armConnections = [
        [11, 13], // hombro izquierdo -> codo izquierdo
        [13, 15], // codo izquierdo -> muñeca izquierda

        [12, 14], // hombro derecho -> codo derecho
        [14, 16]  // codo derecho -> muñeca derecha
    ];

    for (const connection of armConnections) {

        const start = points[connection[0]];
        const end = points[connection[1]];

        if (!start || !end) continue;

        drawLine(
            start.x * canvas.width,
            start.y * canvas.height,
            end.x * canvas.width,
            end.y * canvas.height
        );

        drawPoint(
            start.x * canvas.width,
            start.y * canvas.height
        );

        drawPoint(
            end.x * canvas.width,
            end.y * canvas.height
        );
    }
}

/*
====================================
LANDMARKS → ARRAY
====================================
*/

function flattenLandmarks(landmarks) {

    let output = [];

    for (const point of landmarks) {

        output.push(point.x);
        output.push(point.y);
        output.push(point.z);
    }

    return output;
}

/*
====================================
FRAME FIJO 126 FEATURES
====================================
*/

function flattenArmLandmarks(poseResults) {

    let output = [];

    const emptyArm =
        Array(FEATURES_POSE_ARM).fill(0);

    if (
        !poseResults ||
        !poseResults.landmarks ||
        poseResults.landmarks.length === 0
    ) {
        return emptyArm;
    }

    const points = poseResults.landmarks[0];

    const armPoints = [
        11, // hombro izquierdo
        13, // codo izquierdo
        15, // muñeca izquierda
        12, // hombro derecho
        14, // codo derecho
        16  // muñeca derecha
    ];

    for (const index of armPoints) {

        const point = points[index];

        if (point) {
            output.push(point.x);
            output.push(point.y);
            output.push(point.z);
        } else {
            output.push(0);
            output.push(0);
            output.push(0);
        }
    }

    return output;
}

function buildFrameData(handResults, poseResults) {

    let frameData = [];

    if (handResults.landmarks[0]) {
        frameData.push(
            ...flattenLandmarks(handResults.landmarks[0])
        );
    } else {
        frameData.push(
            ...Array(FEATURES_PER_HAND).fill(0)
        );
    }

    if (handResults.landmarks[1]) {
        frameData.push(
            ...flattenLandmarks(handResults.landmarks[1])
        );
    } else {
        frameData.push(
            ...Array(FEATURES_PER_HAND).fill(0)
        );
    }

    frameData.push(
        ...flattenArmLandmarks(poseResults)
    );

    frameData = frameData.slice(0, TOTAL_FEATURES);

    while (frameData.length < TOTAL_FEATURES) {
        frameData.push(0);
    }

    return frameData;
}

/*
====================================
GUARDAR DATASET
====================================
*/

function saveDataset() {

    const label =
        labelInput.value.trim();

    const data = {

        label: label,

        frames: recordedFrames
    };

    const blob =
        new Blob(
            [JSON.stringify(data)],
            {
                type: "application/json"
            }
        );

    const url =
        URL.createObjectURL(blob);

    const a =
        document.createElement("a");

    a.href = url;

    a.download =
        `${label}_${Date.now()}.json`;

    a.click();

    URL.revokeObjectURL(url);

    recordedFrames = [];
}

/*
====================================
PREDICCIÓN IA
====================================
*/

async function predictSign() {

    if (
        predictionBuffer.length <
        FRAMES_TO_RECORD
    ) {

        return;
    }

    try {

        const response =
            await fetch(
                "https://kinesv-production.up.railway.app/predict",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({
                        frames: predictionBuffer
                    })
                }
            );

        const data =
            await response.json();

        predictionElement.innerText =
            data.prediction;

        confidenceElement.innerText =
            `Confianza: ${(data.confidence * 100).toFixed(1)}%`;

    } catch (error) {

        console.error(error);
    }
}

/*
====================================
RENDER LOOP
====================================
*/

async function render() {

    canvas.width =
        video.videoWidth;

    canvas.height =
        video.videoHeight;

    async function frame() {

        ctx.drawImage(
    video,
    0,
    0,
    canvas.width,
    canvas.height
);

// Detectar pose (brazos)

const poseResults =
    poseLandmarker.detectForVideo(
        video,
        performance.now()
    );

if (
    poseResults.landmarks &&
    poseResults.landmarks.length > 0
) {
    drawPoseArm(poseResults.landmarks);
}

// Detectar manos
const results =
    handLandmarker.detectForVideo(
        video,
        performance.now()
    );

        /*
        ============================
        MANOS DETECTADAS
        ============================
        */

        if (
            results.landmarks &&
            results.landmarks.length > 0
        ) {

            /*
            DIBUJAR MANOS
            */

            for (const landmarks of results.landmarks) {

                drawHand(landmarks);
            }

            /*
            GENERAR FRAME
            */

            const frameData =
                buildFrameData(results, poseResults);

            /*
            BUFFER IA
            */

            predictionBuffer.push(frameData);

            if (
                predictionBuffer.length >
                FRAMES_TO_RECORD
            ) {

                predictionBuffer.shift();
            }

            /*
            PREDICCIÓN
            */

            predictSign();

            /*
            GRABAR DATASET
            */

            if (recording) {

                recordedFrames.push(frameData);

                predictionElement.innerText =
                    `Grabando ${recordedFrames.length}/${FRAMES_TO_RECORD}`;

                confidenceElement.innerText =
                    "";

                if (
                    recordedFrames.length >=
                    FRAMES_TO_RECORD
                ) {

                    recording = false;

                    saveDataset();

                    predictionElement.innerText =
                        "Dataset guardado";
                }
            }

        } else {

            predictionElement.innerText =
                "Esperando manos...";

            confidenceElement.innerText =
                "";
        }

        requestAnimationFrame(frame);
    }

    frame();
}

/*
====================================
INIT
====================================
*/

async function init() {

    try {

        await setupCamera();

        await createHandLandmarker();

        await createPoseLandmarker();

        render();

    } catch (error) {

        console.error(error);

        predictionElement.innerText =
            "Error iniciando cámara";
    }
}

init();