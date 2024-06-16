document.addEventListener("DOMContentLoaded", function() {
    const calculateButton = document.getElementById("calculate");
    const tooltip = document.getElementById("fill-fields-tooltip");
    const formFields = Array.from(document.querySelectorAll("#calculate-form input"));

    formFields.forEach(field => {
        field.addEventListener("input", function() {
            const allFilled = formFields.every(input => input.value.trim() !== "");
            calculateButton.disabled = !allFilled;
            tooltip.style.display = allFilled ? 'none' : 'inline-block';
        });
    });

    calculateButton.addEventListener("click", function() {
        const data = {
            unemployment: document.getElementById("unemployment").value,
            employment: document.getElementById("employment").value,
            potential_labor_force: document.getElementById("potential_labor_force").value,
            salary: document.getElementById("salary").value,
            education_school: document.getElementById("education_school").value,
            education_high: document.getElementById("education_high").value,
            crimes: document.getElementById("crimes").value,
            life_quality: document.getElementById("life_quality").value,
            house_afford: document.getElementById("house_afford").value,
            vrp_2023: document.getElementById("vrp_2023").value,
            population: document.getElementById("population").value,
        };

        fetch("/calculate", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(data)
        })
        .then(response => response.json())
        .then(result => {
            document.getElementById("result-class").innerText = `Класс: ${result.class}`;
            document.getElementById("result-reason").innerText = `Причина: ${result.reason}`;
        })
        .catch(error => console.error("Ошибка:", error));
    });
});