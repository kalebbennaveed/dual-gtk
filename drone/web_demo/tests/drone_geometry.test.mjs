import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
const asModule = async path => import('data:text/javascript;base64,' + Buffer.from(await readFile(new URL(path,import.meta.url))).toString('base64'));
const THREE = await asModule('../vendor/three.module.js');
const { createDroneGeometry } = await asModule('../drone_geometry.js');

test('every transformed airframe vertex lies inside its collision sphere', () => {
  for (const radius of [.05, .1, .4]) {
    const group = createDroneGeometry(THREE, radius, '#159c75');
    assert(group.userData.unscaledRadius > .15);
    group.position.set(8,8,1.3);
    group.rotation.set(.2,-.3,1.7);
    group.updateMatrixWorld(true);
    let maximum = 0;
    group.traverse(mesh => {
      const attribute = mesh.geometry?.attributes.position;
      if (!attribute) return;
      for (let i=0;i<attribute.count;i++) {
        const p = new THREE.Vector3().fromBufferAttribute(attribute,i).applyMatrix4(mesh.matrixWorld).sub(group.position);
        maximum = Math.max(maximum,p.length());
        assert(p.length() < radius, `vertex outside physical radius ${radius}`);
      }
    });
    assert(maximum > .99*radius);
  }
});

test('invalid physical radii are rejected', () => {
  for (const radius of [0,-1,NaN]) assert.throws(() => createDroneGeometry(THREE,radius,'#000000'));
});
