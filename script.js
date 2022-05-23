let utk_data, utk_header, chart

let utk = {
  Corruption: [
    "gaussian-noise",
    "shot-noise",
    "impulse-noise",
    "defocus-blur",
    "glass-blur",
    "motion-blur",
    "zoom-blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic-transform",
    "pixelate",
    "jpeg-compression",
  ],
  Service: ["AWS", "Azure", "GCP"],
  Age: ["0-18", "19-45", "45-64", "65+"],
  Gender: ["Female", "Male"],
  Severity: ["1", "2", "3", "4", "5"],
}

function load_utk(data) {
  utk_data = data.replaceAll('"', "").split(/\r?\n|\r/)
  utk_header = utk_data[0].split(",")
  create_utk_plot()
}

function process_buttons() {
  // get what buttons are turned on
  return ["#age", "#gender", "#service", "#corruption", "#severity"].map(
    (selector) => document.querySelector(selector).checked
  )
}

function get_x_axis_utk(
  age_checked,
  gender_checked,
  service_checked,
  corruption_checked,
  severity_checked
) {
  // determine what the x axis will be
  if (age_checked) return ["Age", utk["Age"].length]
  if (gender_checked) return ["Gender", utk["Gender"].length]
  if (service_checked) return ["Service", utk["Service"].length]
  if (corruption_checked) return ["Corruption", utk["Corruption"].length]
  if (severity_checked) return ["Severity", utk["Severity"].length]
  return ["", 1]
}

function create_utk_plot() {
  let ds = utk
  let ds_header = utk_header

  let [
    age_checked,
    gender_checked,
    service_checked,
    corruption_checked,
    severity_checked,
  ] = process_buttons()
  let [x_axis, x_length] = get_x_axis_utk(
    age_checked,
    gender_checked,
    service_checked,
    corruption_checked,
    severity_checked
  )

  let columns = []
  if (age_checked && x_axis !== "Age") columns.push("Age")
  if (gender_checked && x_axis !== "Gender") columns.push("Gender")
  if (service_checked && x_axis !== "Service") columns.push("Service")
  if (corruption_checked && x_axis !== "Corruption") columns.push("Corruption")
  if (severity_checked && x_axis !== "Severity") columns.push("Severity")

  array_of_names = columns.map((item) => ds[item])

  if (array_of_names.length === 0) array_of_names = [["data"]]

  lines = array_of_names.reduce((a, b) =>
    a.reduce((r, v) => r.concat(b.map((w) => [].concat(v, w))), [])
  )
  console.log(lines, array_of_names)
  data = {}
  for (let i = 0; i < lines.length; i++) {
    console.log(lines[i])
    data[lines[i]] = {
      metric: Array(x_length).fill([0]).flat(),
      count: Array(x_length).fill([0]).flat(),
    }
  }
  console.log(data)
  for (let singleRow = 1; singleRow < utk_data.length - 1; singleRow++) {
    let rowCells = utk_data[singleRow].split(",")
    let colnames = []
    for (let i = 0; i < columns.length; i++) {
      colnames.push(rowCells[ds_header.indexOf(columns[i])])
    }
    if (colnames.length === 0) {
      colnames = ["data"]
    }
    if (x_axis !== "") {
      let xi = ds_header.indexOf(x_axis)
      let x = ds[x_axis].indexOf(rowCells[xi])
      data[colnames.join(",")]["metric"][x] += parseFloat(rowCells[6])
      data[colnames.join(",")]["count"][x] += parseFloat(rowCells[7])
    } else {
      data["data"]["metric"][0] += parseFloat(rowCells[6])
      data["data"]["count"][0] += parseFloat(rowCells[7])
    }
  }
  console.log(data)

  let out_array = []
  Object.entries(data).forEach(([key, val]) => {
    out_array.push(
      [key].concat(
        val["metric"].map(function (item, index) {
          return item / val["count"][index]
        })
      )
    )
  })

  chart = c3.generate({
    bindto: "#chart",
    data: {
      columns: out_array,
    },
    axis: {
      x: {
        type: "category",
        categories: ds[x_axis],
      },
    },
    padding: { right: 40 },
    title: { text: "UTKFace" },
    tooltip: {
      format: {
        value: d3.format(".4f"),
      },
    },
  })
}

function updateView() {
  chart.destroy()
  create_utk_plot()
}

fetch("./data/web_utk_ap075.csv")
  .then((resp) => resp.text())
  .then(load_utk)

document.querySelectorAll("input[type='checkbox']").forEach((checkbox) => {
  checkbox.addEventListener("click", updateView)
})
