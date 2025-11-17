using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;

public class CourseAgent : Agent
{
    [Header("Scene refs")]
    public Transform goal;
    public Transform ground;

    [Header("Movement")]
    public float moveSpeed = 4f;
    public float turnSpeed = 180f;
    public float jumpForce = 5f;

    Rigidbody _rb;
    float _prevDist;

    public override void Initialize()
    {
        _rb = GetComponent<Rigidbody>();
        _rb.maxAngularVelocity = 20f;
    }

    public override void OnEpisodeBegin()
    {
        // Reset agent
        _rb.linearVelocity = Vector3.zero;
        _rb.angularVelocity = Vector3.zero;
        transform.localPosition = new Vector3(Random.Range(-8f, -6f), 1.0f, Random.Range(-2f, 2f));
        transform.localRotation = Quaternion.Euler(0f, Random.Range(0f, 360f), 0f);

        // Randomize goal position on the opposite side
        goal.localPosition = new Vector3(Random.Range(6f, 8f), 0.5f, Random.Range(-2f, 2f));

        _prevDist = Vector3.Distance(transform.localPosition, goal.localPosition);
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        Vector3 toGoal = (goal.localPosition - transform.localPosition);
        sensor.AddObservation(transform.InverseTransformDirection(_rb.linearVelocity)); // 3
        sensor.AddObservation(toGoal.normalized);                                 // 3
        sensor.AddObservation(toGoal.magnitude / 20f);                            // 1 (normalized distance)
        sensor.AddObservation(transform.forward);                                 // 3
        // RayPerceptionSensor handles environment context; no code needed here.
    }

    // Actions: [0]=moveX [-1..1], [1]=moveZ [-1..1], [2]=turn [-1..1], [3]=jumpGate [-1..1]
    public override void OnActionReceived(ActionBuffers actions)
    {
        var ca = actions.ContinuousActions;
        float moveX = Mathf.Clamp(ca[0], -1f, 1f);
        float moveZ = Mathf.Clamp(ca[1], -1f, 1f);
        float turn  = Mathf.Clamp(ca[2], -1f, 1f);
        float jumpGate = ca[3];

        // Move in agent's local frame
        Vector3 move = (transform.right * moveX + transform.forward * moveZ) * moveSpeed;
        _rb.AddForce(new Vector3(move.x, 0f, move.z), ForceMode.Acceleration);

        // Turn
        transform.Rotate(Vector3.up, turn * turnSpeed * Time.fixedDeltaTime);

        // Simple jump
        if (IsGrounded() && jumpGate > 0.5f)
            _rb.AddForce(Vector3.up * jumpForce, ForceMode.VelocityChange);

        // Rewards: progress towards goal + tiny time penalty
        float dist = Vector3.Distance(transform.localPosition, goal.localPosition);
        float progress = _prevDist - dist;
        AddReward(0.02f * progress);  // progress shaping
        AddReward(-0.001f);           // time penalty
        _prevDist = dist;

        // Terminate if fall
        if (transform.localPosition.y < -1f)
        {
            AddReward(-0.2f);
            EndEpisode();
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var ca = actionsOut.ContinuousActions;
        ca[0] = Input.GetAxis("Horizontal"); // A/D
        ca[1] = Input.GetAxis("Vertical");   // W/S
        ca[2] = Input.GetKey(KeyCode.Q) ? -1f : (Input.GetKey(KeyCode.E) ? 1f : 0f);
        ca[3] = Input.GetKey(KeyCode.Space) ? 1f : -1f;
    }

    bool IsGrounded()
    {
        return Physics.Raycast(transform.position + Vector3.up * 0.1f, Vector3.down, out _, 0.2f);
    }

    void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag("Goal"))
        {
            AddReward(1.0f);
            EndEpisode();
        }
    }

    void OnCollisionEnter(Collision col)
    {
        if (col.collider.CompareTag("Hazard"))
            AddReward(-0.05f);
    }
}