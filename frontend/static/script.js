async function carregarFilmes() {
    const container = document.getElementById("filmes-container");

    try {
        const resposta = await fetch("/filmes");
        const filmes = await resposta.json();

        filmes.forEach(filme => {
            const card = document.createElement("div");
            card.classList.add("card");

            card.innerHTML = `
                <h2>${filme.titulo}</h2>
                <p><strong>Gênero:</strong> ${filme.genero}</p>
                <p><strong>Nota:</strong> ${filme.nota}</p>
            `;

            container.appendChild(card);
        });
    } catch (erro) {
        container.innerHTML = "<p>Erro ao carregar filmes.</p>";
        console.error("Erro:", erro);
    }
}

document.addEventListener("DOMContentLoaded", carregarFilmes);
