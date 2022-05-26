let all_data = {}

let common = {
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
  Severity: ["1", "2", "3", "4", "5"],
}

let utk = {
  Age: ["0-18", "19-45", "45-64", "65+"],
  Gender: ["Female", "Male"],
}

let miap = {
  Age: ["Young", "Middle", "Older", "Unknown"],
  Gender: ["Predominantly Feminine", "Predominantly Masculine", "Unknown"],
}

let ccd = {
  Age: ["0-18", "19-45", "45-64", "65+"],
  Gender: ["Feminine", "Masculine", "Other"],
}

let adience = {
  Age: ["0-2", "3-7", "8-14", "15-24", "25-35", "36-45", "46-59", "60+"],
  Gender: ["Female", "Male"],
}

let metadata = {
  adience: {
    column_names: { ...common, ...adience },
    chart_name: "#adience_chart",
    chart_title: "Adience",
    metric_idx: 6,
    count_idx: 7,
  },
  ccd: {
    column_names: { ...common, ...ccd },
    chart_name: "#ccd_chart",
    chart_title: "CCD",
    metric_idx: 8,
    count_idx: 9,
  },
  miap: {
    column_names: { ...common, ...miap },
    chart_name: "#miap_chart",
    chart_title: "MIAP",
    metric_idx: 6,
    count_idx: 7,
  },
  utk: {
    column_names: { ...common, ...utk },
    chart_name: "#utk_chart",
    chart_title: "UTKFace",
    metric_idx: 6,
    count_idx: 7,
  },
}

function load_dataset_builder(dataset) {
  function load_dataset(data) {
    all_data[dataset] = { data: data.replaceAll('"', "").split(/\r?\n|\r/) }
    all_data[dataset]["header"] = all_data[dataset]["data"][0].split(",")
    create_plot(dataset)
  }
  return load_dataset
}

function process_buttons() {
  // get what buttons are turned on
  return ["#age", "#gender", "#service", "#corruption", "#severity"].map(
    (selector) => document.querySelector(selector).checked
  )
}

function get_x_axis(
  metadata,
  age_checked,
  gender_checked,
  service_checked,
  corruption_checked,
  severity_checked
) {
  // determine what the x axis will be
  if (age_checked) return ["Age", metadata["Age"].length]
  if (gender_checked) return ["Gender", metadata["Gender"].length]
  if (service_checked) return ["Service", metadata["Service"].length]
  if (corruption_checked) return ["Corruption", metadata["Corruption"].length]
  if (severity_checked) return ["Severity", metadata["Severity"].length]
  return ["", 1]
}

function create_plot(dataset) {
  let ds = metadata[dataset]["column_names"]
  let ds_header = all_data[dataset]["header"]
  let ds_data = all_data[dataset]["data"]
  let chart_name = metadata[dataset]["chart_name"]
  let chart_title = metadata[dataset]["chart_title"]
  let metric_idx = metadata[dataset]["metric_idx"]
  let count_idx = metadata[dataset]["count_idx"]

  let [
    age_checked,
    gender_checked,
    service_checked,
    corruption_checked,
    severity_checked,
  ] = process_buttons()
  let [x_axis, x_length] = get_x_axis(
    ds,
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
  // console.log(lines, array_of_names)
  data = {}
  for (let i = 0; i < lines.length; i++) {
    // console.log(lines[i])
    data[lines[i]] = {
      metric: Array(x_length).fill([0]).flat(),
      count: Array(x_length).fill([0]).flat(),
    }
  }
  // console.log(data)
  for (let singleRow = 1; singleRow < ds_data.length - 1; singleRow++) {
    let rowCells = ds_data[singleRow].split(",")
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
      data[colnames.join(",")]["metric"][x] += parseFloat(rowCells[metric_idx])
      data[colnames.join(",")]["count"][x] += parseFloat(rowCells[count_idx])
    } else {
      data["data"]["metric"][0] += parseFloat(rowCells[metric_idx])
      data["data"]["count"][0] += parseFloat(rowCells[count_idx])
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

  all_data[dataset]["chart"] = c3.generate({
    bindto: chart_name,
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
    title: { text: chart_title },
    tooltip: {
      format: {
        value: d3.format(".4f"),
      },
    },
  })
}

function updateView() {
  Object.keys(metadata).forEach((dataset) => {
    all_data[dataset]["chart"].destroy()
    create_plot(dataset)
  })
}

Object.keys(metadata).forEach((dataset) => {
  fetch(`./data/web_${dataset}_ap.csv`)
    .then((resp) => resp.text())
    .then(load_dataset_builder(dataset))
})

document.querySelectorAll("input[type='checkbox']").forEach((checkbox) => {
  checkbox.addEventListener("click", updateView)
})
