export function distance(x1, y1, x2, y2) {
    return Math.hypot(x2 - x1, y2 - y1);
}

export function midpoint(x1, y1, x2, y2) {
    return { x: (x1 + x2) / 2, y: (y1 + y2) / 2 };
}

export function angleBetween(x1, y1, x2, y2, offset = 0) {
    return Math.atan2(y2 - y1, x2 - x1) + offset;
}
