export function setupModal(){
    const modal = document.getElementById("infoModal");
    const content = document.getElementById("modalContent");
    document.getElementById("closeModal").onclick = ()=> modal.classList.add("hidden");

    return {
        show(text){
            content.innerText = text;
            modal.classList.remove("hidden");
        }
    };
}
