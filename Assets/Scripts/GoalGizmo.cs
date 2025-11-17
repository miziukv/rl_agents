using UnityEngine;
public class GoalGizmo : MonoBehaviour
{
    public float radius = 0.5f;
    void OnDrawGizmos() { Gizmos.color = Color.yellow; Gizmos.DrawWireSphere(transform.position, radius); }
}
