document.addEventListener("DOMContentLoaded", function() {
    var width = 800,
        height = 600;

    var svg = d3.select("#map")
                .append("svg")
                .attr("width", width)
                .attr("height", height);

    var projection = d3.geoMercator()
                       .center([105, 57])
                       .scale(500);

    var path = d3.geoPath()
                 .projection(projection);

    var tooltip = d3.select("#tooltip");

    d3.json("https://raw.githubusercontent.com/jetka89/geojson-ressources/master/russia.json", function(error, data) {
        if (error) throw error;

        var regions = topojson.feature(data, data.objects.russia).features;

        svg.selectAll(".region")
           .data(regions)
           .enter().append("path")
           .attr("class", "region")
           .attr("d", path)
           .style("fill", function(d) {
               var regionData = data.find(function(region) {
                   return region.region_code === d.properties.REGION_CODE;
               });
               if (regionData) {
                   return color(regionData.investment_score);
               }
               return "#ccc";
           })
           .on("mouseover", function(event, d) {
               var regionData = data.find(function(region) {
                   return region.region_code === d.properties.REGION_CODE;
               });
               if (regionData) {
                   tooltip.classed("hidden", false)
                          .style("left", (event.pageX + 10) + "px")
                          .style("top", (event.pageY - 28) + "px");
                   tooltip.select("#region-name").text(regionData.region_name);
                   tooltip.select("#investment-score").text(regionData.investment_score);
                   tooltip.select("#investment-reason").text(regionData.investment_reason);
               }
           })
           .on("mouseout", function() {
               tooltip.classed("hidden", true);
           });
    });

    var color = d3.scaleLinear()
                  .domain([1, 7])
                  .range(["green", "red"]);
});
