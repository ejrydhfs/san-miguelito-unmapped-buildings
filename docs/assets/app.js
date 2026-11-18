
const COLORS = {verified:'#c92a2a', probable:'#e8590c', unconfirmed:'#868e96'};
const map = L.map('map', {preferCanvas:true, zoomControl:true});
window.__map = map;
const hash = new URLSearchParams(location.hash.slice(1) || '');
map.setView([Number(hash.get('lat')) || 9.048, Number(hash.get('lon')) || -79.494],
            Number(hash.get('z')) || 13);
map.on('moveend', () => {
  const c = map.getCenter();
  history.replaceState(null, '', `#z=${map.getZoom()}&lat=${c.lat.toFixed(5)}&lon=${c.lng.toFixed(5)}`);
});
const esri = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
  {maxZoom:19, attribution:'Imagery © Esri, Maxar, Earthstar Geographics'}).addTo(map);
const osmBase = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',
  {maxZoom:19, attribution:'© OpenStreetMap contributors'});
L.control.layers({'Satellite (Esri)':esri,'OSM map':osmBase},null,{position:'topright'}).addTo(map);
L.control.scale({imperial:false}).addTo(map);

const groups = {};
const loaded = {};
const loading = html => `<span style="color:#868e96">${html}…</span>`;

function popup(p){
  const cls = {verified:'#c92a2a', probable:'#e8590c', unconfirmed:'#868e96'}[p.v] || '#555';
  const cv = p.c == null ? '–' : Number(p.c).toFixed(2);
  const origin = p.s === 'cv'
    ? '<div>found by the model only</div>'
    : `<div>uncovered part of a ${Number(p.m).toFixed(0)} m² detection
         (${Math.round(100*Number(p.u))} % of it is unmapped)</div>`;
  const near = p.d == null ? '' :
    (Number(p.d) <= 1.5 ? '<div>adjacent to a mapped building</div>'
                        : `<div>${Number(p.d).toFixed(0)} m from the nearest mapped building</div>`);
  return `<div><span class="pill" style="background:${cls}">${p.v}</span>
    <b>${Number(p.a).toFixed(0)} m²</b></div>${origin}
    <div>imagery coverage: ${cv}</div>${near}
    <div><a target="_blank" rel="noopener"
      href="https://www.openstreetmap.org/edit?editor=id#map=19/${p.lat}/${p.lon}">open in iD editor ↗</a></div>`;
}

async function loadLayer(name){
  if (loaded[name]) return groups[name];
  const file = (name === 'grid' || name === 'osm') ? name : `candidates_${name}`;
  const res = await fetch(`data/${file}.json`);
  if (!res.ok) throw new Error(`${file}.json: HTTP ${res.status}`);
  const gj = await res.json();
  if (name === 'grid'){
    const maxC = Math.max(...gj.features.map(f => f.properties.count));
    groups.grid = L.geoJSON(gj, {style: f => ({color:'#6741d9', weight:1,
      fillColor:'#6741d9', fillOpacity:0.05 + 0.28*f.properties.count/maxC}),
      onEachFeature: (f,l) => l.bindTooltip(`${f.properties.count} unmapped · ${f.properties.area_m2} m²`)});
  } else if (name === 'osm'){
    groups.osm = L.geoJSON(gj, {style: () => ({color:'#1c7ed6', weight:1, fill:false})});
  } else {
    groups[name] = L.geoJSON(gj, {
      style: () => styleFor(name),
      onEachFeature: (f,l) => { f.properties.v = name; l.bindPopup(popup(f.properties)); }});
  }
  loaded[name] = true;
  return groups[name];
}

function styleFor(name){
  const z = map.getZoom();
  return {color: COLORS[name], weight:1,
          fillOpacity: name === 'unconfirmed' ? 0.08 : (z >= 16 ? 0.22 : 0.10)};
}
map.on('zoomend', () => {
  for (const k of ['verified','probable','unconfirmed'])
    if (loaded[k]) groups[k].setStyle(() => styleFor(k));
});

const checkboxes = [...document.querySelectorAll('input[data-layer]')];
async function sync(){
  for (const cb of checkboxes){
    const name = cb.dataset.layer;
    const want = cb.checked;
    if (want){
      let g;
      try { g = await loadLayer(name); } catch (e) { console.warn(e); continue; }
      if (!map.hasLayer(g)) g.addTo(map);
    } else if (loaded[name] && map.hasLayer(groups[name])){
      map.removeLayer(groups[name]);
    }
  }
}
checkboxes.forEach(cb => cb.onchange = sync);

(async () => {
  const st = await (await fetch('data/stats.json')).json();
  document.getElementById('sub').textContent =
    `${st.district_short || st.district} · ${st.imagery}`;
  document.getElementById('kpis').innerHTML = `
    <span class="big"><b>${st.candidates.toLocaleString()}</b></span><span class="big">unmapped buildings</span>
    <span>OSM has</span><b>${st.osm_buildings.toLocaleString()}</b>
    <span>detected in total</span><b>${st.ml_buildings.toLocaleString()}</b>
    <span>OSM coverage</span><b>${(100*st.osm_buildings/st.ml_buildings).toFixed(0)}%</b>
    <span>unmapped area</span><b>${(st.area_m2/1e6).toFixed(1)} km²</b>`;
  for (const [k,v] of Object.entries(st.verdicts)){
    const el = document.querySelector(`[data-c="${k}"]`);
    if (el) el.textContent = v.toLocaleString();
  }
  const grid = await (await fetch('data/grid.json')).json();
  const key = {}; grid.features.forEach(f => key[f.properties.grid] = f);
  document.getElementById('cells').innerHTML = grid.features.slice(0,12).map(f =>
    `<div class="cell" data-g="${f.properties.grid}"><span>cell ${f.properties.grid}</span>
      <span>${f.properties.count}</span></div>`).join('');
  document.querySelectorAll('#cells .cell').forEach(el => el.onclick = () => {
    const g = key[el.dataset.g];
    if (g) map.fitBounds(L.geoJSON(g).getBounds());
  });
  await sync();
})();
