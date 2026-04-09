async function send() {
    const text = document.getElementById("input").value;

    const res = await fetch("http://127.0.0.1:8000/query", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({text})
    });

    const data = await res.json();
    alert(JSON.stringify(data));
}
