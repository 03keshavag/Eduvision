export function startQuiz(show) {
    const facts = [
        "Metal atoms lose electrons and become positive ions.",
        "Free electrons form a 'sea' around metal ions.",
        "This sea of electrons allows metals to conduct electricity.",
        "Heating increases electron movement, cooling slows it."
    ];
    show(facts[Math.floor(Math.random()*facts.length)]);
}
