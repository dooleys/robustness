let utk_data, utk_header
let miap_data, miap_header
let ccd_data, ccd_header
let adience_data, adience_header

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

let miap = {
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
  Age: ["Young", "Middle", "Older", "Unknown"],
  Gender: ["Predominantly Feminine", "Predominantly Masculine", "Unknown"],
  Severity: ["1", "2", "3", "4", "5"],
}

let ccd = {
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
  Gender: ["Feminine", "Masculine", "Other"],
  Severity: ["1", "2", "3", "4", "5"],
}

let adience = {
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
  Age: ["0-2", "3-7", "8-14", "15-24", "25-35", "36-45", "46-59","60+"],
  Gender: ["Female", "Male"],
  Severity: ["1", "2", "3", "4", "5"],
}


function load_utk(data) {
  utk_data = data.replaceAll('"', "").split(/\r?\n|\r/)
  utk_header = utk_data[0].split(",")
  create_plot('utk', utk_data, utk_header)
}

function load_miap(data) {
  miap_data = data.replaceAll('"', "").split(/\r?\n|\r/)
  miap_header = miap_data[0].split(",")
  create_plot('miap', miap_data, miap_header)
}

function load_ccd(data) {
  ccd_data = data.replaceAll('"', "").split(/\r?\n|\r/)
  ccd_header = ccd_data[0].split(",")
  create_plot('ccd', ccd_data, ccd_header)
}

function load_adience(data) {
  adience_data = data.replaceAll('"', "").split(/\r?\n|\r/)
  adience_header = adience_data[0].split(",")
  create_plot('adience', adience_data, adience_header)
}

function process_buttons() {
  // get what buttons are turned on
  return ["#age", "#gender", "#service", "#corruption", "#severity"].map(
    (selector) => document.querySelector(selector).checked
  )
}

function get_x_axis(metadata,
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

let metadata = {
  'adience': {
    'column_names': adience,
    'chart_name': '#adience_chart',
    'chart_title': 'Adeice',
    'metric_idx': 6,
    'count_idx': 7,
  },
  'ccd': {
    'column_names': ccd,
    'chart_name': '#ccd_chart',
    'chart_title': 'CCD',
    'metric_idx': 8,
    'count_idx': 9,
  },
  'miap': {
    'column_names': miap,
    'chart_name': '#miap_chart',
    'chart_title': 'MIAP',
    'metric_idx': 6,
    'count_idx': 7,
  },
  'utk': {
    'column_names': utk,
    'chart_name': '#utk_chart',
    'chart_title': 'UTKFace',
    'metric_idx': 6,
    'count_idx': 7,
  }
}

function create_plot(dataset, data, header) {

  let ds = metadata[dataset]['column_names']
  let ds_header = header
  let ds_data = data
  let chart_name = metadata[dataset]['chart_name']
  let chart_title = metadata[dataset]['chart_title']
  let metric_idx = metadata[dataset]['metric_idx']
  let count_idx = metadata[dataset]['count_idx']

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

  c3.generate({
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
  document.getElementById('utk_chart').innerHTML = ''
  document.getElementById('miap_chart').innerHTML = ''
  document.getElementById('ccd_chart').innerHTML = ''
  document.getElementById('adience_chart').innerHTML = ''
  create_plot('utk', utk_data, utk_header)
  create_plot('miap', miap_data, miap_header)
  create_plot('ccd', ccd_data, ccd_header)
  create_plot('adience', adience_data, adience_header)
}

fetch("./data/web_utk_ap.csv")
  .then((resp) => resp.text())
  .then(load_utk);

fetch("./data/web_miap_ap.csv")
  .then((resp) => resp.text())
  .then(load_miap);

fetch("./data/web_ccd_ap.csv")
  .then((resp) => resp.text())
  .then(load_ccd);

fetch("./data/web_adience_ap.csv")
  .then((resp) => resp.text())
  .then(load_adience);

document.querySelectorAll("input[type='checkbox']").forEach((checkbox) => {
  checkbox.addEventListener("click", updateView)
})
