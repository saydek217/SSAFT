using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SpinMouse : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {

    }

    public float speed = 50.0f;
    // Update is called once per frame
    private void Update()
    {

        transform.Rotate(0, speed * Time.deltaTime, 0 , Space.Self);
    }
}
