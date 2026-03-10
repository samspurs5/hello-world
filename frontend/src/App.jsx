import { useState, useEffect } from 'react'
import { MapContainer, TileLayer, Polyline, useMap } from 'react-leaflet'
import 'leaflet/dist/leaflet.css'

const COLORS = ['#e63946', '#2a9d8f', '#e9c46a', '#f4a261', '#a8dadc', '#7400b8']

const DEFAULTS = {
  lat_col: 'lat',
  lon_col: 'lon',
  accuracy_col: '',
  timestamp_col: 'timestamp',
  group_col: '',
  outlier_col: '',
  default_accuracy_m: '10',
  max_time_gap: '300',
  min_segment_length: '3',
}

function FitBounds({ coords }) {
  const map = useMap()
  useEffect(() => {
    if (coords.length > 0) map.fitBounds(coords, { padding: [40, 40] })
  }, [coords])
  return null
}

function buildSegments(records, latCol, lonCol) {
  const map = new Map()
  for (const r of records) {
    const key = `${r.group_key}__${r.segment_index}`
    if (!map.has(key)) map.set(key, { raw: [], filtered: [], groupKey: String(r.group_key) })
    const s = map.get(key)
    if (r[latCol] != null && r[lonCol] != null) s.raw.push([r[latCol], r[lonCol]])
    if (r.filtered_lat != null && r.filtered_lon != null) s.filtered.push([r.filtered_lat, r.filtered_lon])
  }
  return [...map.values()]
}

export default function App() {
  const [cfg, setCfg] = useState(DEFAULTS)
  const [file, setFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const set = (name) => (e) => setCfg((c) => ({ ...c, [name]: e.target.value }))

  const submit = async (e) => {
    e.preventDefault()
    if (!file) return
    setLoading(true)
    setError(null)
    setResult(null)
    const fd = new FormData()
    fd.append('file', file)
    fd.append('config', JSON.stringify(cfg))
    try {
      const r = await fetch('/api/run', { method: 'POST', body: fd })
      const text = await r.text()
      let d
      try { d = JSON.parse(text) } catch { throw new Error(text.slice(0, 200)) }
      if (!r.ok) throw new Error(d.detail || 'Server error')
      setResult(d)
    } catch (err) {
      setError(err.message)
    }
    setLoading(false)
  }

  const segments = result ? buildSegments(result.records, result.lat_col, result.lon_col) : []
  const allCoords = segments.flatMap((s) => s.raw)

  return (
    <div style={{ display: 'flex', height: '100vh', fontFamily: 'system-ui, sans-serif', fontSize: 13 }}>
      {/* ── Sidebar ── */}
      <aside style={S.side}>
        <h2 style={S.title}>kalmangps</h2>

        <form onSubmit={submit} style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          <label>
            <div style={S.label}>CSV file</div>
            <input type="file" accept=".csv" onChange={(e) => setFile(e.target.files[0])} style={{ color: '#cdd6f4' }} />
          </label>

          {[
            ['lat_col',       'Latitude column'],
            ['lon_col',       'Longitude column'],
            ['accuracy_col',  'Accuracy column (m, optional)'],
            ['timestamp_col', 'Timestamp column'],
            ['group_col',     'Group column (optional)'],
            ['outlier_col',   'Outlier flag column (optional)'],
          ].map(([name, label]) => (
            <label key={name}>
              <div style={S.label}>{label}</div>
              <input value={cfg[name]} onChange={set(name)} style={S.input} />
            </label>
          ))}

          <label>
            <div style={S.label}>Default accuracy (m, when no column)</div>
            <input type="number" value={cfg.default_accuracy_m} onChange={set('default_accuracy_m')} style={S.input} />
          </label>
          <label>
            <div style={S.label}>Max time gap (s)</div>
            <input type="number" value={cfg.max_time_gap} onChange={set('max_time_gap')} style={S.input} />
          </label>
          <label>
            <div style={S.label}>Min segment length</div>
            <input type="number" value={cfg.min_segment_length} onChange={set('min_segment_length')} style={S.input} />
          </label>

          <button type="submit" disabled={!file || loading} style={S.btn}>
            {loading ? 'Running…' : 'Run pipeline'}
          </button>
        </form>

        {error && <p style={S.error}>{error}</p>}

        {result && (
          <div style={S.stats}>
            <div style={{ color: '#a6e3a1', marginBottom: 8 }}>
              ✓ {result.n_segments} segment{result.n_segments !== 1 ? 's' : ''} · {result.n_points} points
            </div>
            <div style={{ color: '#6c7086', marginBottom: 4 }}>Legend</div>
            <LegendRow color="#6c7086" dashed label="Raw GPS" />
            {segments.map((s, i) => (
              <LegendRow
                key={i}
                color={COLORS[i % COLORS.length]}
                label={`Segment ${i + 1}${s.groupKey && s.groupKey !== 'None' ? ` · ${s.groupKey}` : ''}`}
              />
            ))}
          </div>
        )}
      </aside>

      {/* ── Map ── */}
      <MapContainer center={[51.5, -0.09]} zoom={13} style={{ flex: 1 }}>
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution='© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
        />
        {segments.map((seg, i) => [
          <Polyline
            key={`raw-${i}`}
            positions={seg.raw}
            pathOptions={{ color: '#6c7086', weight: 1.5, dashArray: '5 5', opacity: 0.7 }}
          />,
          <Polyline
            key={`filtered-${i}`}
            positions={seg.filtered}
            pathOptions={{ color: COLORS[i % COLORS.length], weight: 3 }}
          />,
        ])}
        {allCoords.length > 0 && <FitBounds coords={allCoords} />}
      </MapContainer>
    </div>
  )
}

function LegendRow({ color, dashed, label }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 4 }}>
      <svg width="24" height="6">
        <line
          x1="0" y1="3" x2="24" y2="3"
          stroke={color}
          strokeWidth={dashed ? 1.5 : 3}
          strokeDasharray={dashed ? '4 3' : undefined}
        />
      </svg>
      <span>{label}</span>
    </div>
  )
}

const S = {
  side:  { width: 275, padding: 16, background: '#1e1e2e', color: '#cdd6f4', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 10, flexShrink: 0 },
  title: { margin: 0, fontSize: 18, color: '#cba6f7' },
  label: { fontSize: 11, color: '#a6adc8', marginBottom: 2 },
  input: { width: '100%', background: '#313244', border: '1px solid #45475a', color: '#cdd6f4', padding: '4px 6px', borderRadius: 4 },
  btn:   { background: '#cba6f7', color: '#1e1e2e', border: 'none', padding: '8px 0', borderRadius: 4, fontWeight: 700, cursor: 'pointer', marginTop: 4, fontSize: 13 },
  error: { color: '#f38ba8', fontSize: 12, margin: 0 },
  stats: { borderTop: '1px solid #313244', paddingTop: 12, fontSize: 12 },
}
