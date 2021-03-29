let create_digit = (i) => {
    let digit = {
        container: document.createElement("div"),
        content: {
            label: document.createElement("div"),
            bar: document.createElement("div"),
        },
    };

    digit.content.label.innerHTML = i;

    digit.container.classList.add("digit-container");
    digit.content.label.classList.add("digit-label");
    digit.content.bar.classList.add("digit-bar");

    digit.container.appendChild(digit.content.bar);
    digit.container.appendChild(digit.content.label);
    digit.container.id = i;

    return digit;
};


let digit_container = document.getElementById("digits");
let digits = [];

for(let i = 0; i < 10; i++) {
    let digit = create_digit(i);
    digit_container.appendChild(digit.container);
    digits.push(digit);
}


let model = null;
let model_loaded = false;

let load_model = async () => {
    model = await tf.loadGraphModel("static/lenet5/model.json");
    model_loaded = true;
};
load_model();

let predict = async (ctx, w, h) => {
    let img = ctx.getImageData(0, 0, w, h);
    let x = tf.browser.fromPixels(img).resizeBilinear([28, 28]).mean(2).reshape([1, 1, 28, 28]);
    let y = await model.predict({"img:0": x}, "Identity:0");
    let label = await tf.argMax(y.squeeze(0), 0).dataSync()[0];
    return label;
};


let cvs = document.getElementById("canvas");
let ctx = cvs.getContext("2d");
let rect = cvs.getBoundingClientRect();

let painting = false;
let pos = { x: 0, y: 0 };

let paint_color = "#ffffffff";
let clear_color = "#000000ff";
let stroke_size = 20;

let clear_cvs = () => {
    ctx.fillStyle = clear_color;
    ctx.fillRect(0, 0, cvs.width, cvs.height);
};
clear_cvs();

let infer = async () => {
    if(model_loaded) {
        let label = await predict(ctx, cvs.width, cvs.height);
        for(let i = 0; i < 10; i++) {
            if(label == i) {
                digits[i].content.bar.classList.add("argmax");
                digits[i].content.label.classList.add("argmax");
            } else {
                digits[i].content.bar.classList.remove("argmax");
                digits[i].content.label.classList.remove("argmax");
            }
        }
    }
}

cvs.addEventListener("mousemove", e => {
    let new_pos = { x: e.clientX - rect.left, y: e.clientY - rect.top };
    
    if(painting) {
        ctx.beginPath();
        ctx.strokeStyle = paint_color;
        ctx.lineJoin = ctx.lineCap = "round";
        ctx.lineWidth = stroke_size;
        ctx.moveTo(new_pos.x, new_pos.y);
        ctx.lineTo(pos.x, pos.y);
        ctx.stroke();
        ctx.closePath();
    }

    pos = new_pos;
});

cvs.addEventListener("mousedown", () => { painting = true; clear_cvs(); });
cvs.addEventListener("mouseup",   () => { painting = false; infer(); });
cvs.addEventListener("mouseout",  () => { painting = false; });