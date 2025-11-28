export function generateElectronPositions(count, radius, angleOffset) {
    const step = (Math.PI * 2) / count;
    const positions = [];

    for (let i = 0; i < count; i++) {
        positions.push({
            x: 250 + radius * Math.cos(angleOffset + i * step),
            y: 250 + radius * Math.sin(angleOffset + i * step)
        });
    }

    return positions;
}

export function randomElement(list) {
    return list[Math.floor(Math.random() * list.length)];
}
