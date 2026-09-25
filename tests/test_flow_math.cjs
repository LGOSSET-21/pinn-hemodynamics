const fs=require('fs'),vm=require('vm'),assert=require('assert');
const source=fs.readFileSync(require('path').join(__dirname,'../visualization/flow_math.js'),'utf8');
const ctx={};vm.runInNewContext(source+';globalThis.math=FlowMath;',ctx);
const M=ctx.math;
// u(t,r)=2t+3r : interpolation exacte, même en dehors des noeuds.
const ts=[0,.5,1],rs=[0,.5,1],values=ts.map(t=>rs.map(r=>2*t+3*r));
assert(Math.abs(M.sample(values,rs,ts,.25,.25)-1.25)<1e-12);
assert.strictEqual(M.sample(values,rs,ts,1,1),5);
// Intégrale en temps : t²+3rt ; empêche de confondre vitesse et déplacement.
const acc=M.cumulative(values,ts);
assert(Math.abs(M.integral(values,acc,rs,ts,.25,.25)-.25)<1e-12);
assert(Math.abs(M.integral(values,acc,rs,ts,1,1)-4)<1e-12);
assert.strictEqual(M.linear(ts,[0,2,4],.25),1);
console.log('Interpolation et déplacement des traceurs : 5 contrôles réussis.');
