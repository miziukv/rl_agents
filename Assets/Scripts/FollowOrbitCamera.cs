using UnityEngine;

public class FollowOrbitCamera : MonoBehaviour
{
    public Transform target;
    public float distance = 6f;
    public float height = 2f;
    public float yawSpeed = 120f;   // mouse X
    public float pitchSpeed = 90f;  // mouse Y
    public float minPitch = -20f;
    public float maxPitch = 60f;
    public float smooth = 10f;
    public bool holdRightMouseToOrbit = true;

    float _yaw, _pitch;

    void Start()
    {
        if (target != null)
        {
            Vector3 toTarget = target.position - transform.position;
            _yaw = Mathf.Atan2(toTarget.x, toTarget.z) * Mathf.Rad2Deg;
        }
    }

    void LateUpdate()
    {
        if (target == null) return;

        bool orbiting = !holdRightMouseToOrbit || Input.GetMouseButton(1);
        if (orbiting)
        {
            _yaw   += Input.GetAxis("Mouse X") * yawSpeed * Time.deltaTime;
            _pitch -= Input.GetAxis("Mouse Y") * pitchSpeed * Time.deltaTime;
            _pitch = Mathf.Clamp(_pitch, minPitch, maxPitch);
        }

        Quaternion rot = Quaternion.Euler(_pitch, _yaw, 0f);
        Vector3 desired = target.position + rot * new Vector3(0f, height, -distance);
        transform.position = Vector3.Lerp(transform.position, desired, 1f - Mathf.Exp(-smooth * Time.deltaTime));
        transform.LookAt(target.position + Vector3.up * 1.0f);
    }
}