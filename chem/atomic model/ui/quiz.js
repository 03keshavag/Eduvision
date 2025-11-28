const questions = [
    { element: "H", answer: "1" },
    { element: "He", answer: "2" },
    { element: "Li", answer: "2,1" },
    { element: "C", answer: "2,4" },
    { element: "O", answer: "2,6" },
    { element: "Ne", answer: "2,8" }
];

let current = 0;

export function startQuiz(showModal) {
    const q = questions[current];
    const userAnswer = prompt(`What is the electron configuration of ${q.element}?`);

    if (!userAnswer) return;

    if (userAnswer.replace(/\s/g,"") === q.answer) {
        showModal(`✅ Correct! ${q.element} = ${q.answer}`);
        current = (current + 1) % questions.length;
    } else {
        showModal(`❌ Incorrect.\nCorrect Answer: ${q.answer}`);
    }
}
