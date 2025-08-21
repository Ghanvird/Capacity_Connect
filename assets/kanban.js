// /* assets/kanban.js */
// window.dash_clientside = window.dash_clientside || {};
// window.dash_clientside.kanban = {
//   scroll: function(nL, nR){
//     const el = document.querySelector('.ws-right-card .ws-kanban');
//     if(!el){ return ''; }

//     // Which button fired?
//     const trig = dash_clientside.callback_context.triggered[0]?.prop_id || '';
//     const dir = trig.indexOf('kanban-right') !== -1 ? 1 : -1;

//     // Scroll exactly one column width per click
//     const col = el.querySelector('.ws-kanban-col');
//     const step = col ? (col.getBoundingClientRect().width + 16) : 320;

//     el.scrollBy({ left: dir * step, behavior: 'smooth' });
//     return ''; // dummy output
//   }
// };
