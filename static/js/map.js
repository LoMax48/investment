document.addEventListener("DOMContentLoaded", function() {
    const svg = d3.select("svg");
    const colorScale = d3.scaleSequential(d3.interpolateRdYlGn).domain([7, 1]);

    svg.selectAll("path")
        .attr("fill", function() {
            return colorScale(d3.select(this).attr("data-result"));
        })
        .on("mouseover", function(event) {
            const region = d3.select(this);

            if (region.attr("data-result") == 1) {
                d3.select("#tooltip")
                    .style("display", "block")
                    .html(`<strong>${region.attr("data-title")}</strong><br>Класс: ${region.attr("data-result")}`);
            } else {
                d3.select("#tooltip")
                    .style("display", "block")
                    .html(`<strong>${region.attr("data-title")}</strong><br>Класс: ${region.attr("data-result")}<br>${region.attr("data-reason")}`);
            }
        })
        .on("mousemove", function(event) {
            d3.select("#tooltip")
                .style("left", (event.pageX + 10) + "px")
                .style("top", (event.pageY + 10) + "px");
        })
        .on("mouseout", function() {
            d3.select("#tooltip").style("display", "none");
        });
});