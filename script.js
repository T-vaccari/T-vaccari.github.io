// script.js

// Funzione per calcolare la somma
function calcolaSomma() {
    // Ottieni i valori inseriti dall'utente
    const numero1 = parseFloat(document.getElementById('numero1').value);
    const numero2 = parseFloat(document.getElementById('numero2').value);
  
    // Verifica se i valori sono numeri validi
    if (!isNaN(numero1) && !isNaN(numero2)) {
      // Calcola la somma
      const somma = numero1 + numero2;
  
      // Mostra il risultato in un alert
      alert("La somma di " + numero1 + " e " + numero2 + " è: " + somma);
    } else {
      // Avvisa l'utente se i valori inseriti non sono numeri validi
      alert("Inserisci due numeri validi.");
    }
  }
  
  // Gestisci l'invio del modulo
  document.getElementById('formNumeri').addEventListener('submit', function (event) {
    event.preventDefault(); // Previeni il comportamento predefinito del modulo
    calcolaSomma(); // Chiamata alla funzione per calcolare la somma
  });
  