const bondingQuestions = [
    { a: "Na", b: "Cl", type: "ionic", answer: "Na loses 1 electron, Cl gains 1." },
    { a: "H", b: "O", type: "covalent", answer: "Share electrons to form H₂O." },
    { a: "C", b: "H", type: "covalent", answer: "Share electrons to form CH₄." }
];

let current = 0;

export function startQuiz(showModal, rebuildAtoms) {
    const q = bondingQuestions[current];
    rebuildAtoms(q.a, q.b);

    setTimeout(() => {
        showModal(`Build bond between ${q.a} and ${q.b}.\nBond Type: ${q.type.toUpperCase()}`);
    }, 500);
}

export function checkQuiz(showModal) {
    const q = bondingQuestions[current];
    showModal(`Correct!\n${q.answer}`);
    current = (current + 1) % bondingQuestions.length;
}
