export function random(min, max) {
    return Math.random() * (max - min) + min;
}

export function distance(x1, y1, x2, y2) {
    return Math.hypot(x2 - x1, y2 - y1);
}

export function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
}

export function bounceWithinBounds(electron, canvas) {
    if (electron.x < 0 || electron.x > canvas.width) electron.speedX *= -1;
    if (electron.y < 0 || electron.y > canvas.height) electron.speedY *= -1;
}
