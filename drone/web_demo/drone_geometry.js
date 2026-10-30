/** The rendered airframe fits the same circumsphere used by collision checks. */
export function createDroneGeometry(THREE, radius, color) {
  if (!(radius > 0)) throw new Error('Positive physical collision radius required');
  const group = new THREE.Group();
  const material = new THREE.MeshStandardMaterial({color, roughness: .45});
  group.add(new THREE.Mesh(new THREE.BoxGeometry(.13, .09, .035), material));
  for (const a of [-1, 1]) for (const b of [-1, 1]) {
    const arm = new THREE.Mesh(new THREE.BoxGeometry(.14, .014, .012), material);
    arm.rotation.z = Math.atan2(b, a);
    arm.position.set(a * .048, b * .048, 0);
    group.add(arm);
    const rotor = new THREE.Mesh(new THREE.TorusGeometry(.037, .008, 8, 20), material);
    rotor.position.set(a * .075, b * .075, .018);
    group.add(rotor);
  }
  group.updateMatrixWorld(true);
  const vertex = new THREE.Vector3();
  let maximumRadius = 0;
  group.traverse(mesh => {
    const position = mesh.geometry?.attributes.position;
    if (!position) return;
    for (let i = 0; i < position.count; i++) {
      vertex.fromBufferAttribute(position, i).applyMatrix4(mesh.matrixWorld);
      maximumRadius = Math.max(maximumRadius, vertex.length());
    }
  });
  // A small margin also covers Float32 vertex/transform roundoff.
  group.scale.setScalar(.995 * radius / maximumRadius);
  group.userData = {collisionRadius: radius, unscaledRadius: maximumRadius,
                    renderedRadius: .995 * radius};
  return group;
}
