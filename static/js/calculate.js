document.addEventListener("DOMContentLoaded", function() {
    const calculateButton = document.getElementById("calculate");
    const formFields = Array.from(document.querySelectorAll("#calculate-form input"));

    formFields.forEach(field => {
        field.addEventListener("input", function() {
            const allFilled = formFields.every(input => input.value.trim() !== "");
            calculateButton.disabled = !allFilled;
        });
    });

    calculateButton.addEventListener("click", function() {
        const data = {
            unemployment: parseFloat(document.getElementById("unemployment").value),
            employment: parseFloat(document.getElementById("employment").value),
            potential_labor_force: parseFloat(document.getElementById("potential_labor_force").value),
            salary: parseFloat(document.getElementById("salary").value),
            education_school: parseFloat(document.getElementById("education_school").value),
            education_high: parseFloat(document.getElementById("education_high").value),
            crimes: parseFloat(document.getElementById("crimes").value),
            life_quality: parseFloat(document.getElementById("life_quality").value),
            house_afford: parseFloat(document.getElementById("house_afford").value),
            vrp_2023: parseFloat(document.getElementById("vrp_2023").value) / parseFloat(document.getElementById("population").value),
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
            if (result.class != 1) {
                document.getElementById("result-reason").innerText = `Рекомендации: ${result.reason}`;
            }
        })
        .catch(error => console.error("Ошибка:", error));
    });
});